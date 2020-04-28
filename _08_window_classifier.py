

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
np.random.seed(4) # the seed should be the batch number
trnotes = np.random.choice(notes_2018, len(notes_2018)*2//3, replace = False)
tenotes = [i for i in notes_2018 if i not in trnotes]
# get a vector of non-negatives for case weights
non_neutral = (np.sum(np.array(y_dums[[i for i in y_dums.columns if "_0" not in i]]), axis = 1)>1).astype('float32')
tr_caseweights = np.array([non_neutral[i] for i in range(len(non_neutral)) if df.note.iloc[i] in trnotes])
nnweight = 1/np.mean(tr_caseweights)
tr_caseweights[tr_caseweights == 1] *= nnweight
tr_caseweights[tr_caseweights == 0] = 1
te_caseweights = np.array([non_neutral[i] for i in range(len(non_neutral)) if df.note.iloc[i] in tenotes])
te_caseweights[te_caseweights == 1] *= nnweight
te_caseweights[te_caseweights == 0] = 1




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
            0, # l1/l2 penalty
            bool(np.random.choice(list(range(2)))))  # semipar
    model = makemodel(*hps)
    return model, hps


loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


# initialize a df for results
hpdf = pd.DataFrame(dict(idx=list(range(100)),
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
for seed in range(100):
    try:
        np.random.seed(seed+4*100) # the seed should always be the batch number * 100 plus the iter
        # shrunk model
        model, hps = draw_hps(seed+4*100)
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
        # yte = [tf.convert_to_tensor(i) for i in yte]
        print("\n\n********************************\n\n")
        print(hpdf.iloc[model_iteration])
        
        tr_caseweights = []
        te_caseweights = []
        for i in range(len(ytr)):
            x = (ytr[i][:,1] == 0).astype('float32')
            wt = 1/np.mean(x)
            x[x == True] *= wt
            x[x == 0] = 1
            tr_caseweights.append(x)
            x = (yte[i][:,1] == 0).astype('float32')
            x[x == True] *= wt
            x[x == 0] = 1
            te_caseweights.append(x)
            
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
        # w = model.get_weights()
        # # set the bias terms to the proportions
        # for i in range(4):
        #     props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
        #                       inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
        #                       inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
        #     # print(props)
        #     pos = 7 - i * 2
        #     # print(pos)
        #     # print(w[-pos].shape)
        #     w[-pos] = w[-pos] * 0 + props
    
        # model.set_weights(w)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
              loss={'Msk_prob':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Nutrition':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Resp_imp':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                    'Fall_risk':tf.keras.losses.CategoricalCrossentropy(from_logits=False)})

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                    patience=20,
                                                    restore_best_weights = True)
        model.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
                  batch_size=256,
                  epochs=1000, 
                  callbacks = [callback],
                  sample_weight = tr_caseweights,
                  validation_data = ([Xte_np, Xte_p], yte, te_caseweights)) if hps[5] is True else (Xte, yte, te_caseweights))
        model.save_weights(f"{outdir}saved_models/model_{seed}_batch_4")
        
        pred = model.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
        # initialize the loss and the optimizer
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
        loss = loss_object(yte, pred)
    
        print(f"at {datetime.datetime.now()}")
        print(f"test loss: {loss}")
    
        tf.keras.backend.clear_session()
        hpdf.loc[model_iteration, 'best_loss'] = float(loss)
        hpdf.loc[model_iteration, 'time_to_convergence'] = time.time() - start_time
        hpdf.to_csv(f"{outdir}hyperparameter_gridsearch_27apr_win.csv")
        model_iteration += 1
    except Exception as e:
        send_message_to_slack(e)
        break


# '''
# Finally, fit and save the actual model used
# '''
# hpdf = pd.read_csv(f"{outdir}hyperparameter_gridsearch_21apr_win.csv")
# best= hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]

# seed = 92
# np.random.seed(seed)
# # shrunk model
# model, hps = draw_hps(seed)
    
# # put the data in arrays for modeling, expanding out to the window size
# # only converting the test into tensors, to facilitate indexing
# if hps[-1] is False: # corresponds with the semipar argument
#     Xtr = tensormaker(sdf, trnotes, embedding_colnames+str_varnames, hps[0])            
#     Xte = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames+str_varnames, hps[0]), dtype = 'float32')
# else:
#     Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])            
#     Xte_np = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames, hps[0]), dtype = 'float32')
#     Xtr_p = np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
#     Xte_p = tf.convert_to_tensor(np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in tenotes]), dtype = 'float32')
# ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
# yte = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))
# yte = [tf.convert_to_tensor(i) for i in yte]

# start_time = time.time()


# # initialize the bias terms with the logits of the proportions
# w = model.get_weights()
# # set the bias terms to the proportions
# for i in range(4):
#     props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
#                       inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
#                       inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
#     pos = 7 - i * 2
#     w[-pos] = w[-pos] * 0 + props

# model.set_weights(w)

# model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
#       loss={'Msk_prob':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#             'Nutrition':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#             'Resp_imp':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
#             'Fall_risk':tf.keras.losses.CategoricalCrossentropy(from_logits=False)})

# callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
#                                             patience=20,
#                                             restore_best_weights = True)
# model.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
#           batch_size=256,
#           epochs=1000, 
#           callbacks = [callback],
#           validation_data = ([Xte_np, Xte_p], yte) if hps[5] is True else (Xte, yte))

# model.save(f"{outdir}saved_AL_models/model_batch_4.h5")
# model = tf.keras.models.load_model(f"{outdir}saved_AL_models/model_batch_4.h5")
# yhat= model.predict([Xte_np, Xte_p])


# np.mean(np.sum(-1*np.array(yte[1])* np.log(yhat[1]), axis = 1))

# np.mean((yte[1].numpy()- yhat[1])**2)
# yhat[1]

# loss_object(yte[1], yhat[1])

# plt.hist(yhat[3][:,1])
# yhat[0][:,1].min()


# np.max(yhat[1][:,1])
# np.min(yhat[1][:,1])

# df.to_csv(f"{outdir}foo.csv")

# '''
# Active learning
# loop through the notes and compute their entropies for all of the phenotypes
# '''
# enotes = os.listdir(f"{outdir}embedded_notes")
# # trim off the ones that are not from 2018
# enotes = [enotes[i] for i in range(len(enotes)) if int(enotes[i].split("_")[2][1:])<=12]
# # trim off the ones that are in the training set already
# screen = [notes_2018[i].split("_")[2]+"_"+notes_2018[i].split("_")[3] for i in range(len(notes_2018))]
# tocheck = [enotes[i].split("_")[2]+"_"+enotes[i].split("_")[3] for i in range(len(enotes))]
# enotes_not_yet_used = [enotes[i] for i in range(len(enotes)) if tocheck[i] not in screen]
# assert len(enotes) > len(enotes_not_yet_used)

# # load the structured data
# strdat = pd.read_csv(f"{outdir}impdat_dums.csv")
# strdat = strdat.drop(columns = "Unnamed: 0")

# i = 0
# # load, merge, scale
# x = pd.read_pickle(f"{outdir}embedded_notes/{enotes_not_yet_used[i]}")
# x = x[[i for i in x.columns if i in embedding_colnames+['PAT_ID', 'month']]]
# x = x.merge(strdat)
# x[embedding_colnames+str_varnames] = scaler.transform(x[embedding_colnames+str_varnames])
# # tensorize it
# x['note'] = "placeholder"
# xnp = tensormaker(x, ['placeholder'], embedding_colnames, hps[0])
# xnp.shape
# xp = np.array(x[str_varnames])
# pred = model.predict([xnp, xp])
# len(pred)
# plt.hist(H(pred[1]))

# H(pred[1]).max()

# w = model.get_weights()
# plt.hist(w[-2].flatten())
# model.summary()

# max(pred[0][:,1])



# def H(p):
#     '''entropy'''
#     return -np.sum(p*np.log(p), axis = 1)
# plt.hist(H(pred[1))

# H(pred[0]).max()


# xp.shape

#     Xte_np = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames, hps[0]), dtype = 'float32')



# # scaling
# scaler = StandardScaler()
# scaler.fit(df[embedding_colnames+str_varnames].loc[df.note.isin(trnotes)])
# sdf = copy.deepcopy(df)
# sdf[embedding_colnames+str_varnames] = scaler.transform(df[embedding_colnames+str_varnames])



# def tensormaker(D, notelist, cols, ws):
#     # take a data frame and a list of notes and a list of columns and a window size and return an array for feeting to tensorflow
#     note_arrays = [np.array(D.loc[D.note == i, cols]) for i in notelist]
#     notelist = []
#     for j in range(len(note_arrays)):
#         lags, leads = [], []
#         for i in range(int(np.ceil(ws/2))-1, 0, -1):
#             li = np.concatenate([np.zeros((i,note_arrays[j].shape[1])), note_arrays[j][:-i]], axis = 0)
#             lags.append(li)
#         assert len(set([i.shape for i in lags])) == 1 # make sure they're all the same size                
#         for i in range(1, int(np.floor(ws/2))+1, 1):
#             li = np.concatenate([note_arrays[j][i:], np.zeros((i,note_arrays[j].shape[1]))], axis = 0)
#             leads.append(li)
#         assert len(set([i.shape for i in leads])) == 1 # make sure they're all the same size                
#         x = np.squeeze(np.stack([lags+ [note_arrays[j]] + leads]))
#         notelist.append(np.swapaxes(x, 1, 0))
#     return np.concatenate(notelist, axis = 0)


