
"""
Start with a few simple models, and compare them:
    - Regular LSTM
    - Bidirectional LSTM
    - 1-D convolutional
    - Dilated convolutional

"""


import os
import pandas as pd
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
import re
import numpy as np
from _99_project_module import nrow, write_txt, ncol
from scipy.stats import norm
import matplotlib.pyplot as plt
import inspect
import sys
import multiprocessing as mp
import platform
import time

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dense, concatenate, Flatten, Conv1D, \
     LSTM, LeakyReLU, BatchNormalization
from tensorflow.keras import Model, Input, backend

outdir = f"{os.getcwd()}/output/"

# load datasets
df = pd.concat([
    pd.read_csv(f"{outdir}batch1_data_ft_oa_corp_300d_bw5.csv"),
    pd.read_csv(f"{outdir}batch2_data_ft_oa_corp_300d_bw5.csv"),
    pd.read_csv(f"{outdir}batch3_data_ft_oa_corp_300d_bw5.csv")
])

#trim the lag off
df = df[[i for i in df.columns if "lag" not in i]]

# load and merge the structured data
strdat = pd.DataFrame(dict(
    month = df.note.apply(lambda x: int(x.split("_")[2][1:])),
    PAT_ID = df.note.apply(lambda x: x.split("_")[3]),
    note = df.note
))

for i in [i for i in os.listdir(outdir) if "_6m" in i]:
    x = pd.read_pickle(f"{outdir}{i}")
    x = x.drop(columns = "PAT_ENC_CSN_ID")
    strdat = strdat.merge(x, how = 'left')
    print(strdat.shape)

# add columns for missing values of structured data
for i in strdat.columns:
    if any(strdat[[i]].isna()):
        strdat[[i+"_miss"]] = strdat[[i]].isna().astype(int)
        strdat[[i]] = strdat[[i]].fillna(0)

# save the column names so that I can find them later
str_varnames = list(strdat.columns[3:])
out_varnames = df.columns[5:10]

# append
assert(len([i for i in range(len(df.note)) if df.note.iloc[i] != strdat.note.iloc[i]])==0)
df = pd.concat([df.reset_index(drop = True), strdat[str_varnames].reset_index(drop = True)], axis = 1)
df.shape

# make output dummies
y_dums = []
for i in out_varnames:
    dum = pd.get_dummies(df[[i]])
    dum.columns

df.Fall_risk.

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
input_timesteps = 1000
input_dims = 600 + len(str_varnames)
out_varnames = df.columns[5:10]
df['month'] = df.note.apply(lambda x: int(x.split("_")[2][1:]))
trnotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) < 13]
tenotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
note_lengths = df.note.value_counts()
embedding_colnames = [i for i in df.columns if re.match("identity|wmean", i)]


# simple bilstm with parametric part for the structured data
input = Input(shape = (input_timesteps, input_dims - len(str_varnames)))
str_input = Input((input_timesteps, len(str_varnames)))
LSTM_forward = LSTM(50, return_sequences = True)(input)
LSTM_backward = LSTM(50, return_sequences = True, go_backwards = True)(input)
LSTM_backward = backend.reverse(LSTM_backward, axes = 1)
conc = concatenate([LSTM_forward, LSTM_backward, str_input], axis = 2)
outlayers = [Dense(3, activation = "softmax", name = i)(conc) for i in out_varnames]
model = Model([input, str_input], outlayers)
model.summary()


def cutter_padder(df, i, cols, samp, input_timesteps):
    di = df.loc[df.note == i, cols]
    x = np.array(di)
    if nrow(x) > input_timesteps:
        x = x[samp:,:][:input_timesteps,:]
    x = np.concatenate([x,np.zeros((input_timesteps-nrow(x), len(cols)))], axis = 0)
    return x

# batchmaker function
def batchmaker(notes_set):
    notes_set = trnotes
    '''
    function to make batches to feed to TF.  Because some notes are longer 
    than others, and to prevent oversampling the middles of notes, this 
    function will randomly cut the note and pad with zeros, even in cases 
    where the note length is greater then the number of inpout timesteps.
    '''
    emlist, strlist, ylist = [], [], []
    for i in notes_set:
        samp = np.random.choice(note_lengths[i])
        # embeddings and structured data
        emlist.append(cutter_padder(df, i, embedding_colnames, samp, input_timesteps))
        strlist.append(cutter_padder(df, i, str_varnames, samp, input_timesteps))
        # y variables
        ycut = cutter_padder(df, i, out_varnames, samp, input_timesteps)
        ycut.shape
        pd.get_dummies(pd.Series(ycut[:,0]))
        ylist.append()

    ydum = [pd.get_dummies(i) for i in ylist]        
    
# training function

# training loop







