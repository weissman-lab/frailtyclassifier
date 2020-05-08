

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

def makemodel(window_size, n_dense, nunits,
              dropout, penalty):
    pen = 10**penalty
    inp = Input(shape=(window_size, len(embedding_colnames)))
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
    outlayers = [Dense(3, activation="softmax", name=i, 
                       kernel_regularizer = l1_l2(pen))(fl)
                 for i in out_varnames]
    model = Model(inp, outlayers)
    return model

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


# datasets
datasets = [i for i in os.listdir(outdir) if 'diagnostic_' in i]
print(datasets)

outlist = []
for ii in range(len(datasets)):
    print(datasets[ii])
    df = pd.read_csv(f"{outdir}{datasets[ii]}")
    df.drop(columns = "Unnamed: 0", inplace = True)
    out_varnames = list(df.columns[5:9])
    y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
    
    # define some useful constants
    embedding_colnames = [i for i in df.columns if re.match("identity", i)]
    input_dims = len(embedding_colnames) 
    notes_2018 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) <= 12]
    notes_2019 = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
    note_lengths = df.note.value_counts()
    np.random.seed(8675309) # the seed should be the batch number
    trnotes = np.random.choice(notes_2018, len(notes_2018)*2//3, replace = False)
    tenotes = [i for i in notes_2018 if i not in trnotes]
    # get a vector of non-negatives for case weights

    
    model = makemodel(window_size = 10, 
                      n_dense = 3,
                      nunits = 50,
                      dropout = .3,
                      penalty = -5)
    
    # initialize the bias terms with the logits of the proportions
    w = model.get_weights()
    # set the bias terms to the proportions
    for i in range(4):
        props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
                          inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
                          inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
        pos = 7 - i * 2
        w[-pos] = w[-pos] * 0 + props
    
    sdf = copy.deepcopy(df)
    
    scaler = StandardScaler()
    scaler.fit(sdf[embedding_colnames].loc[df.note.isin(trnotes)])
    sdf[embedding_colnames] = scaler.transform(sdf[embedding_colnames])
    sdf = pd.concat([sdf, y_dums], axis = 1)
    
    Xtr = tensormaker(sdf, trnotes, embedding_colnames, 10)            
    Xval = tensormaker(sdf, tenotes, embedding_colnames, 10)            
    
    ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
    yval = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))
    
    tr_caseweights = []
    for i in range(len(ytr)):
        x = (ytr[i][:,1] == 0).astype('float32')
        wt = 1/np.mean(x)
        x[x == True] *= wt
        x[x == 0] = 1
        tr_caseweights.append(x)
        
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
          loss={'Msk_prob':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                'Nutrition':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                'Resp_imp':tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                'Fall_risk':tf.keras.losses.CategoricalCrossentropy(from_logits=False)})
    
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                patience=10,
                                                restore_best_weights = True)
    model.fit(Xtr, ytr,
              batch_size=256,
              epochs=1000, 
              callbacks = [callback],
              sample_weight = tr_caseweights,
              validation_data = (Xval, yval),
              verbose = 0)
    model.save(f"{outdir}diagnostic_model_may2020_DS{datasets[ii]}.h5")
    
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 
    pred = model.predict(Xval)
    # catprop
    catprop = np.mean([np.mean(x[:,1]) for x in pred]    )
    best_loss = loss_object(yval, pred)
    
    out = dict(embedding = datasets[ii],
               best_loss = float(best_loss),
               catprop = catprop)
    print(out)
    outlist.append(out)
    print(pd.DataFrame(outlist))
    
outdf = pd.DataFrame(outlist)
outdf.to_csv(f"{outdir}diagnostic_may2020_result.csv")

# compare against counterfactual
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False) 

w = model.get_weights()
for i in range(4):
    props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
                      inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
                      inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
    pos = 7 - i * 2
    print(pos)
    w[-pos] = w[-pos] * 0 + props
    w[-(pos+1)] = w[-(pos+1)] * 0 
    
model.set_weights(w)
cfac = model.predict(Xval)
cfac_loss = loss_object(yval, cfac)
catprop = np.mean([np.mean(x[:,1]) for x in cfac])




