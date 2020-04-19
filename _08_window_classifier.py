

import os
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
if 'crandrew' in os.getcwd():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from _99_project_module import inv_logit, send_message_to_slack
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate, Conv1D, \
    LeakyReLU, BatchNormalization, LSTM, Dropout, Dense, Flatten
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Model, Input, backend
import pickle
import time
from sklearn.preprocessing import StandardScaler
import copy

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"


if 'moddat.pkl' not in os.listdir(outdir):
    # import structured data 
    strdat = pd.read_csv(f"{outdir}impdat_dums.csv")
    strdat = strdat.drop(columns = "Unnamed: 0")
    
    # load the annotations
    df = pd.concat([
        pd.read_csv(f"{outdir}batch1_data_ft_oa_corp_300d_bw5.csv"),
        pd.read_csv(f"{outdir}batch2_data_ft_oa_corp_300d_bw5.csv"),
        pd.read_csv(f"{outdir}batch3_data_ft_oa_corp_300d_bw5.csv")
    ])
    # this below is a fix to a naming error that I made when I created batch 4
    # it's only for clarity:  the train/test split is done based on the month
    df4 = pd.read_csv(f"{outdir}batch4_data_ft_oa_corp_300d_bw5.csv")
    df4.note = df4.note.str.replace("batch_03", "batch_04")
    df = pd.concat([df, df4])

    # trim the lag off
    df = df[[i for i in df.columns if ("lag" not in i) and ("wmean" not in i)]]
    
    # create patient ID and month variables
    tm = pd.DataFrame(dict(month=df.note.apply(lambda x: int(x.split("_")[2][1:])),
                           PAT_ID=df.note.apply(lambda x: x.split("_")[3])))
    df = pd.concat([tm, df], axis = 1)
    df = df.drop(columns = "Unnamed: 0")
    
    # merge on the structured data
    df = df.merge(strdat, how = 'left')
    
    str_varnames = list(strdat.columns[2:])
    out_varnames = df.columns[7:11]
    
    y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
    df = pd.concat([y_dums, df], axis=1)
    moddat = dict(df=df,
                  str_varnames=str_varnames,
                  out_varnames=out_varnames,
                  y_dums=y_dums)
    pickle.dump(moddat, open(f"{outdir}moddat.pkl", "wb"))
else:
    moddat = pickle.load(open(f"{outdir}moddat.pkl", "rb"))
    df = moddat['df']
    str_varnames = moddat['str_varnames']
    out_varnames = moddat['out_varnames']
    y_dums = moddat['y_dums']
    del moddat


# define some useful constants
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
input_dims = len(embedding_colnames) + len(str_varnames)
notes_2018 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) <= 12]
notes_2019 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
note_lengths = df.note.value_counts()
trnotes = np.random.choice(notes_2018, len(notes_2018)*2//3, replace = False)
tenotes = [i for i in notes_2018 if i not in trnotes]


def makemodel(window_size, n_dense, nunits,
              dropout, penalty, semipar):
    pen = 10**penalty
    if semipar is True:
        base_shape = input_dims - len(str_varnames)
        top_shape = input_dims - len(embedding_colnames)
    else:
        base_shape = input_dims
    inp = Input(shape=(window_size, base_shape))
    LSTM_forward = LSTM(nunits, return_sequences = True, 
                        kernel_regularizer = l1_l2(pen))(inp)
    LSTM_backward = LSTM(nunits, return_sequences = True, go_backwards = True, 
                         kernel_regularizer = l1_l2(pen))(inp)
    LSTM_backward = backend.reverse(LSTM_backward, axes = 1)
    conc = concatenate([LSTM_forward, LSTM_backward], axis = 2)
    # dense
    for i in range(n_dense):
        d = Dense(nunits, kernel_regularizer = l1_l2(pen))(conc if i == 0 else drp)
        lru = LeakyReLU()(d)
        drp = Dropout(dropout)(lru)
    fl = Flatten()(drp)
    if semipar is True:
        p_inp = Input(shape = (top_shape))
        conc = concatenate([p_inp, fl])
    outlayers = [Dense(3, activation="softmax", name=i, 
                       kernel_regularizer = l1_l2(pen))(conc if semipar is True else fl)
                 for i in out_varnames]
    if semipar is True:
        model = Model([inp, p_inp], outlayers)
    else: 
        model = Model(inp, outlayers)
    return model
    



def draw_hps(seed):
    np.random.seed(seed)
    hps = (int(np.random.choice(list(range(4, 40)))),  # window size
           int(np.random.choice(list(range(1, 10)))),  # n dense
           int(np.random.choice(list(range(10, 100)))),  # n units
           float(np.random.uniform(low = 0, high = .5)),  # dropout
           float(np.random.uniform(low = -8, high = -1)), # l1/l2 penalty
           bool(np.random.choice(list(range(2)))))  # semipar
    model = makemodel(*hps)
    return model, hps




loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# initialize a df for results
hpdf = pd.DataFrame(dict(idx=list(range(1000)),
                         oob = np.nan,
                         window_size=np.nan,
                         n_dense=np.nan,
                         n_units=np.nan,
                         dropout=np.nan,
                         l1_l2=np.nan,
                         semipar=np.nan,
                         time_to_convergence=np.nan,
                         best_loss=np.nan))


def tensormaker(D, notelist, cols, ws):
    # take a data frame and a list of notes and a list of columns and a window size and return an array for feeting to tensorflow
    note_arrays = [np.array(D.loc[D.note == i, cols]) for i in notelist]
    notelist = []
    for j in range(len(note_arrays)):
        lags, leads = [], []
        for i in range(int(np.ceil(ws/2))-1, 0, -1):
            li = np.concatenate([np.zeros((i,note_arrays[j].shape[1])), note_arrays[j][:-i]], axis = 0)
            lags.append(li)
        assert len(set([i.shape for i in lags])) == 1 # make sure they're all the same size                
        for i in range(1, int(np.floor(ws/2))+1, 1):
            li = np.concatenate([note_arrays[j][i:], np.zeros((i,note_arrays[j].shape[1]))], axis = 0)
            leads.append(li)
        assert len(set([i.shape for i in leads])) == 1 # make sure they're all the same size                
        x = np.squeeze(np.stack([lags+ [note_arrays[j]] + leads]))
        notelist.append(np.swapaxes(x, 1, 0))
    return np.concatenate(notelist, axis = 0)

def make_y_list(y):
    return [y[:, i * 3:(i + 1) * 3] for i in range(len(out_varnames))]


# scaling
scaler = StandardScaler()
scaler.fit(df[embedding_colnames+str_varnames].loc[df.note.isin(trnotes)])
sdf = copy.deepcopy(df)
sdf[embedding_colnames+str_varnames] = scaler.transform(df[embedding_colnames+str_varnames])



model_iteration = 0
for seed in range(1000):
    try:
        np.random.seed(seed)
        # shrunk model
        model, hps = draw_hps(seed)
        for i in range(2, 8): # put the hyperparameters in the hpdf
            hpdf.loc[model_iteration, hpdf.columns[i]] = hps[i - 2]
        hpdf.loc[model_iteration, 'oob'] = ",".join(tenotes)
            
        # put the data in arrays for modeling, expanding out to the window size
        # only converting the test into tensors, to facilitate indexing
        if hps[-1] is False: # corresponds with the semipar argument
            Xtr = tensormaker(sdf, trnotes, embedding_colnames+str_varnames, hps[0])            
            Xte = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames+str_varnames, hps[0]), dtype = 'float32')
        else:
            Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])            
            Xte_np = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames, hps[0]), dtype = 'float32')
            Xtr_p = np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
            Xte_p = tf.convert_to_tensor(np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in tenotes]), dtype = 'float32')
        ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
        yte = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))
        yte = [tf.convert_to_tensor(i) for i in yte]
        print("\n\n********************************\n\n")
        print(hpdf.iloc[model_iteration])
        
        start_time = time.time()
    
        # # initial overfit
        # ofm = makemodel(hps[0], hps[1], hps[2], 0, 10**-np.inf, hps[5])
        # ofm.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
        #               loss={'Msk_prob':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        #                     'Nutrition':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        #                     'Resp_imp':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        #                     'Fall_risk':tf.keras.losses.CategoricalCrossentropy(from_logits=False)})
        

        # ofm.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
        #         batch_size=256,
        #         epochs=100)
        # #transfer    
        # model.set_weights(ofm.get_weights())
        
        # initialize the bias terms with the logits of the proportions
        w = model.get_weights()
        # set the bias terms to the proportions
        for i in range(4):
            props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
                              inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
                              inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
            # print(props)
            pos = 7 - i * 2
            # print(pos)
            # print(w[-pos].shape)
            w[-pos] = w[-pos] * 0 + props
    
        model.set_weights(w)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss={'Msk_prob':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Nutrition':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Resp_imp':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Fall_risk':tf.keras.losses.CategoricalCrossentropy(from_logits=False)})

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=5,
                                                    restore_best_weights = True)
        model.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
                  batch_size=256,
                  epochs=1000, 
                  callbacks = [callback],
                  validation_data = ([Xte_np, Xte_p], yte) if hps[5] is True else (Xte, yte))
        
        pred = model.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
        # initialize the loss and the optimizer
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
        loss = loss_object(yte, pred)
    
        print(f"at {datetime.datetime.now()}")
        print(f"test loss: {loss}")
    
        tf.keras.backend.clear_session()
        hpdf.loc[model_iteration, 'best_loss'] = float(loss)
        hpdf.loc[model_iteration, 'time_to_convergence'] = time.time() - start_time
        hpdf.to_csv(f"{outdir}hyperparameter_gridsearch_19apr_win.csv")
        model_iteration += 1
    except Exception as e:
        send_message_to_slack(e)
        break







# # training function
# @tf.function()
# def train(x, y):
#     with tf.GradientTape() as g:
#         pred = model(x)
#         loss = loss_object(y, pred)
#         gradients = g.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(gradients, model.trainable_variables))


# def test():
#     lossvec = []
#     wtvec = []
#     for i in tenotes:
#         tebatch = batchmaker(i)
#         pred = model(tebatch['x'])
#         lossvec.append(loss_object(tebatch['y'], pred))
#         wtvec.append(note_lengths[i])
#     avg_loss = sum([float(lossvec[i]) * wtvec[i] for i in range(len(lossvec))]) / sum(wtvec)
#     return avg_loss



# model_iteration = 0
# for seed in range(1000):
#     try:
#         np.random.seed(seed)

#         model, hps = draw_hps(seed)
#         for i in range(1, 7):
#             hpdf.loc[model_iteration, hpdf.columns[i]] = hps[i - 1]

#         print("\n\n********************************\n\n")
#         print(hpdf.iloc[model_iteration])

#         start_time = time.time()

#         # initialize the bias terms with the logits of the proportions
#         w = model.get_weights()

#         for i in range(5):
#             props = np.array([inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == -1)),
#                               inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == 0)),
#                               inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == 1))])
#             # print(props)
#             pos = 9 - i * 2
#             # print(pos)
#             # print(w[-pos].shape)
#             w[-pos] = w[-pos] * 0 + props

#         model.set_weights(w)

#         loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
#         optimizer = tf.keras.optimizers.Adam(learning_rate=.00005)


#         # training function
#         @tf.function(experimental_relax_shapes=True)
#         def train(x, y):
#             with tf.GradientTape() as g:
#                 pred = model(x)
#                 loss = loss_object(y, pred)
#                 gradients = g.gradient(loss, model.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, model.trainable_variables))


#         def test():
#             lossvec = []
#             wtvec = []
#             for i in tenotes:
#                 tebatch = batchmaker(i)
#                 pred = model(tebatch['x'])
#                 lossvec.append(loss_object(tebatch['y'], pred))
#                 wtvec.append(note_lengths[i])
#             avg_loss = sum([float(lossvec[i]) * wtvec[i] for i in range(len(lossvec))]) / sum(wtvec)
#             return avg_loss


#         # training loop
#         osrmse = np.empty((10000, 5))
#         oslossvec = []
#         test_rate = 75
#         stopcounter = 0
#         best = 9999
#         iter = 0

#         while stopcounter < 5:
#             if iter % test_rate == 0:
#                 osloss = test()
#                 oslossvec.append(osloss)
#                 if osloss < best:
#                     best = osloss
#                     bestidx = iter
#                     stopcounter = 0
#                 else:
#                     stopcounter += 1
#                 print(f"at {datetime.datetime.now()}")
#                 print(f"test loss: {osloss}, "
#                       f"stopcounter: {stopcounter}, "
#                       f"iter: {iter // test_rate}")
#             train(batch['x'], batch['y'])
#             iter += 1

#         tf.keras.backend.clear_session()
#         hpdf.loc[model_iteration, 'best_loss'] = best
#         hpdf.loc[model_iteration, 'time_to_convergence'] = time.time() - start_time
#         hpdf.to_csv(f"{outdir}hyperparameter_gridsearch_results.csv")
#         model_iteration += 1
#     except Exception as e:
#         send_message_to_slack(e)
#         break






# def cutter_padder(df, note, cols):
#     di = df.loc[df.note == note, cols]
#     x = np.array(di).astype('float32')
#     return x


# # batchmaker function
# def batchmaker(note, return_y_as_list=True, batchsize = 256):
#     emdat = cutter_padder(df, note, embedding_colnames)
#     strdat = cutter_padder(df, note, str_varnames)
#     X = np.concatenate([emdat, strdat], axis=1)
#     # y variables
#     yvars = cutter_padder(df, note, y_dums.columns)
#     if return_y_as_list == True:
#         # break out the outcomes into a list of tensors
#         y_list_of_tensors = [tf.convert_to_tensor(yvars[:, i * 3:(i + 1) * 3]) for i in range(5)]
#         output = dict(x=X,
#                       y=y_list_of_tensors)
#     else:
#         output = dict(x=X,
#                       y=yvars)
#     return output

# # make all batches for a given window size
# mbx, mby = [], []
# for j in range(len(trnotes)):
#     b = batchmaker(trnotes[j], return_y_as_list=False)
#     for i in range(0, b['x'].shape[0], window_size):
#         xi = b['x'][i:(i + 20), :]
#         yi = b['y'][i:(i + 20), :]
#         if xi.shape[0] < 20:
#             xi = np.concatenate([xi, np.zeros((window_size-xi.shape[0], xi.shape[1]))], axis = 0)
#             yi = np.concatenate([yi, np.zeros((window_size-yi.shape[0], yi.shape[1]))], axis = 0)
#         mbx.append(xi)
#         mby.append(yi)

# X = tf.convert_to_tensor(np.stack(mbx))
# Y = np.stack(mby)



# @@expand dims and otherwise clean up



        # def batchmaker(idx, semipar):
        #     if semipar is True:
        #         x = [tf.convert_to_tensor(Xtr_np[idx,:,:], dtype = 'float32'), 
        #              tf.convert_to_tensor(Xtr_p[idx,:], dtype = 'float32')]
        #     else:
        #         x = tf.convert_to_tensor(Xtr[idx,:,:], dtype = 'float32')
        #     y = [tf.convert_to_tensor(i[idx, :]) for i in ytr]
        #     return dict(x=x, y=y)
