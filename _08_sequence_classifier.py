

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
    LeakyReLU, BatchNormalization
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
    # trim the lag off
    df = df[[i for i in df.columns if ("lag" not in i) and ("wmean" not in i)]]
    
    # create patient ID and month variables
    tm = pd.DataFrame(dict(month=df.note.apply(lambda x: int(x.split("_")[2][1:])),
                           PAT_ID=df.note.apply(lambda x: x.split("_")[3])))
    df = pd.concat([tm, df], axis = 1)
    df = df.drop(columns = "Unnamed: 0")
    
    # merge on the structured data
    df = df.merge(strdat, how = 'left')
    
    str_varnames = list(strdat.columns[3:])
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



'''
Model ideas (simple to more complicated):
    - Simple LSTM
    - Simple 1d conv
    - Deeper versions of both above
    - MT language model that also does sequence prediction
'''

'''
Issue: notes have varying lengths.  Possible solutions:
    --batch size of 1, feed to LSTMs in native resolution
    --batch size of N, chunk to 1000 (padding the short ones)
    
    Problem with first approach is that it's incompatible with anything besides lstms.
    Problem with second approach is in testing:  boundary effects
    Second approach seems better on balance.
'''
# define some useful constants
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
input_dims = len(embedding_colnames) + len(str_varnames)
notes_2018 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) <= 12]
notes_2019 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
note_lengths = df.note.value_counts()


# nlayers = 10
# nfilters = 100
# kernel_size = 10
# out_kernel_size = 3
# batch_normalization = True
# half_dilated = True


def makemodel(nlayers, nfilters, kernel_size, out_kernel_size, batch_normalization):
    inp = Input(shape=(None, input_dims))
    lay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same')(inp)
    lay = LeakyReLU()(lay)
    if batch_normalization == True:
        lay = BatchNormalization()(lay)
    for i in range(nlayers):
        lay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same')(lay)
        lay = LeakyReLU()(lay)
        if batch_normalization == True:
            lay = BatchNormalization()(lay)
    outlayers = [Conv1D(filters=3, kernel_size=out_kernel_size, activation="softmax", padding='same', name=i)(lay)
                 for i in out_varnames]
    model = Model(inp, outlayers)
    return model



def draw_hps(seed):
    np.random.seed(seed)
    hps = (int(np.random.choice(list(range(2, 21)))),  # n layers
           int(np.random.choice(list(range(10, 201)))),  # n filters
           int(np.random.choice(list(range(5, 21)))),  # kernel_size
           int(np.random.choice(list(range(1, 10)))),  # output kernel size
           bool(np.random.choice(list(range(2)))))  # batch normalization
    model = makemodel(*hps)
    return model, hps



def cutter_padder(df, note, cols):
    '''
    take a data frame and a note and a set of columns are return an array for tensorflow
    '''
    di = df.loc[df.note == note, cols]
    x = np.array(di).astype('float32')
    return x


# batchmaker function
def batchmaker(df, note, return_y_as_list=True):
    '''
    One batch = one note given that this script is implementing a sequence model and the notes are different lengths.  
    The input is an array of the embeddings and the output is (optionally) a list of tensors for the four categories
    '''
    emdat = cutter_padder(df, note, embedding_colnames)
    strdat = cutter_padder(df, note, str_varnames)
    X = np.concatenate([emdat, strdat], axis=1)
    # y variables
    yvars = cutter_padder(df, note, y_dums.columns)
    if return_y_as_list == True:
        # break out the outcomes into a list of tensors
        y_list_of_tensors = [tf.convert_to_tensor(yvars[:, i * 3:(i + 1) * 3]) for i in range(4)]
        output = dict(x=tf.convert_to_tensor(np.expand_dims(X, 0)),
                      y=y_list_of_tensors)
    else:
        output = dict(x=tf.convert_to_tensor(np.expand_dims(X, 0)),
                      y=yvars)
    return output


# initialize a df for results
hpdf = pd.DataFrame(dict(idx=list(range(1000)),
                         oob = np.nan,
                         nlayers=np.nan,
                         nfilters=np.nan,
                         kernel_size=np.nan,
                         out_kernel_size=np.nan,
                         batch_normalization=np.nan,
                         time_to_convergence=np.nan,
                         best_loss=np.nan))


model_iteration = 0
for seed in range(1000):
    try:
        np.random.seed(seed)
        
        inbag = np.random.choice(list(range(len(notes_2018))), 
                                              replace = True, 
                                              size = len(notes_2018))
        trnotes = [notes_2018[i] for i in inbag]
        tenotes = [i for i in notes_2018 if i not in trnotes]

        # scaling
        scaler = StandardScaler()
        scaler.fit(df[embedding_colnames+str_varnames].loc[df.note.isin(trnotes)])
        sdf = copy.deepcopy(df)
        sdf[embedding_colnames+str_varnames] = scaler.transform(df[embedding_colnames+str_varnames])
        
        model, hps = draw_hps(seed)
        for i in range(2, 7): # put the hyperparameters in the hpdf
            hpdf.loc[model_iteration, hpdf.columns[i]] = hps[i - 2]

        print("\n\n********************************\n\n")
        print(hpdf.iloc[model_iteration])

        start_time = time.time()

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
        
        # initialize the loss and the optimizer
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        optimizer = tf.keras.optimizers.Adam(learning_rate=.00005)

        # training function
        @tf.function(experimental_relax_shapes=True)
        def train(x, y):
            with tf.GradientTape() as g:
                pred = model(x)
                loss = loss_object(y, pred)
                gradients = g.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        def test():
            lossvec = []
            wtvec = []
            for i in tenotes:
                tebatch = batchmaker(sdf, i)
                pred = model(tebatch['x'])
                lossvec.append(loss_object(tebatch['y'], pred))
                wtvec.append(note_lengths[i])
            avg_loss = sum([float(lossvec[i]) * wtvec[i] for i in range(len(lossvec))]) / sum(wtvec)
            return avg_loss


        # training loop
        osrmse = np.empty((10000, 4))
        oslossvec = []
        test_rate = 50 # check against the test set after how many iterations?
        stopcounter = 0
        best = 9999
        iter = 0

        while stopcounter < 20:
            if iter % test_rate == 0:
                osloss = test()
                oslossvec.append(np.log(osloss))
                if osloss < best:
                    best = osloss
                    bestidx = iter
                    stopcounter = 0
                else:
                    stopcounter += 1
                print(f"at {datetime.datetime.now()}")
                print(f"test loss: {osloss}, "
                      f"stopcounter: {stopcounter}, "
                      f"iter: {iter // test_rate}")
            batch = batchmaker(sdf, trnotes[np.random.choice(len(trnotes))])
            train(batch['x'], batch['y'])
            iter += 1

        tf.keras.backend.clear_session()
        hpdf.loc[model_iteration, 'best_loss'] = best
        hpdf.loc[model_iteration, 'time_to_convergence'] = time.time() - start_time
        hpdf.to_csv(f"{outdir}hyperparameter_gridsearch_10apr_seq.csv")
        model_iteration += 1
    except Exception as e:
        send_message_to_slack(e)
        break




# def makeplot():
#     fig, ax = plt.subplots(figsize=(24, 9), ncols=3, nrows=1, sharex=False)
#     ax[0].plot(list(range(len(oslossvec))), oslossvec, label='test rmse')
#     ax[0].plot(list(range(len(oslossvec))), [np.mean(oslossvec[:i][-20:]) for i in len(oslossvec)])
#     ax[0].set_ylabel('Loss')
#     ax[0].set_xlabel('iteration // test_rate')
#     ax[0].grid('on')
#     ax[0].axhline(y=best, color='r', linestyle='-')
#     for i in range(5):
#         ax[1].plot(list(range(len(oslossvec))), osrmse[:iter // test_rate + 1, i], label=out_varnames[i])
#     ax[1].set_ylabel('RMSE')
#     ax[1].set_xlabel('iteration // test_rate')
#     ax[1].grid('on')
#     ax[1].legend()
#     for i in range(5):
#         ax[2].plot(list(range(len(oslossvec))), rmse_rare[:iter // test_rate + 1, i], label=out_varnames[i])
#     ax[2].set_ylabel('RMSE, non-neutral')
#     ax[2].set_xlabel('iteration // test_rate')
#     ax[2].grid('on')
#     ax[2].legend()
#     fig.savefig(f"{figdir}LSTM_mar26.png")
#     plt.close('all')


# def get_rare_rmse(y, pred):
#     rmsevec = []
#     for i in range(len(y)):
#         idx = np.where(y[i][:, 1] == 0)
#         if len(idx[0]) == 0:
#             rmsevec.append(np.nan)
#         else:
#             br = y[i].numpy()[idx[0], :]
#             pr = np.squeeze(pred[i])[idx[0], :]
#             rmsevec.append(np.mean((br - pr) ** 2) ** .5)
#     return rmsevec
