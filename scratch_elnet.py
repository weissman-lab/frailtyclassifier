#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 11:46:37 2020

@author: crandrew
"""


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
sdf = copy.deepcopy(df)

def labsmooth(x):
    for i in range(len(sdf.Nutrition)):
        out = copy.deepcopy(sdf.Nutrition)
        dn = i-5 if i>5 else 0
        up = i+5 if i<len(sdf.Nutrition) else len(sdf.Nutrition)
        sp = sdf.Nutrition[dn:up]
        if all(sp == 0):
            pass
        elif len(set(sp))==3:
            vv = np.mean([j for j in sp if j !=0])
            out[i] = vv
        elif len(set(sp)) == 2:
            out[i] = list(set(sp[sp!=0]))[0]
    return out
        
for v in out_varnames:
    sdf[v] = labsmooth(sdf[v])

non_neutral = np.sum(y_dums[[i for i in y_dums.columns if "_0" not in i]], axis = 1)>1
nndf = sdf.loc[non_neutral]
nndf = pd.concat([nndf for i in range(int(len(sdf)/len(nndf)))])
sdf = pd.concat([sdf, nndf])

scaler = StandardScaler()
scaler.fit(sdf[embedding_colnames+str_varnames].loc[df.note.isin(trnotes)])
sdf[embedding_colnames+str_varnames] = scaler.transform(sdf[embedding_colnames+str_varnames])



Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, 10)            
Xte_np = tf.convert_to_tensor(tensormaker(sdf, tenotes, embedding_colnames, 10), dtype = 'float32')
Xtr_p = np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
Xte_p = tf.convert_to_tensor(np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in tenotes]), dtype = 'float32')

ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
yte = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))

xn = Xtr_np.reshape(Xtr_np.shape[0], Xtr_np.shape[1]* Xtr_np.shape[2])
Xtr = np.concatenate([xn, Xtr_p], axis = 1)
xn = Xte_np.numpy().reshape(Xte_np.shape[0], Xte_np.shape[1]* Xte_np.shape[2])
Xte = np.concatenate([xn, Xte_p], axis = 1)

yr = sdf.loc[sdf.note.isin(trnotes), out_varnames]
ye = sdf.loc[sdf.note.isin(tenotes), out_varnames]


# dftr = pd.concat([yr, pd.DataFrame(Xtr, columns = embedding_colnames+str_varnames)], axis = 1)
# dfte = pd.concat([ye, pd.DataFrame(Xte, columns = embedding_colnames+str_varnames)], axis = 1)


from sklearn.linear_model import LogisticRegression

def f(lam):
    clf = LogisticRegression(random_state=0, max_iter = 10000, 
                             penalty = "elasticnet",
                             solver = "saga",
                             l1_ratio = .5,
                             C = 1/(10**lam),
                             multi_class = "multinomial").fit(Xtr, yr.Nutrition)
    pred = clf.predict_proba(Xte)
    vloss = np.mean(-np.sum(yte[1] * np.log(pred), axis = 1))
    print(f"reg: 10^{lam}, vloss: {vloss}")
    return dict(m = clf, loss = vloss)
    

lams = list(range(-10, 6))
import multiprocessing as mp
pool = mp.Pool(mp.cpu_count())
reslist = pool.map(f, lams)
pool.close()

with open(f'{outdir}reslist.pkl', 'wb') as handle:
    pickle.dump(reslist, handle, protocol=pickle.HIGHEST_PROTOCOL)


