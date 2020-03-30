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
if 'crandrew' in os.getcwd():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from _99_project_module import inv_logit, send_message_to_slack
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, concatenate, Flatten, Conv1D, \
    LSTM, LeakyReLU, BatchNormalization
from tensorflow.keras import Model, Input, backend
import pickle
import time

outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"
embedded_outdir = f"{outdir}embedded_notes/"

if 'moddat.pkl' not in os.listdir(outdir):
    # load datasets
    df = pd.concat([
        pd.read_csv(f"{outdir}batch1_data_ft_oa_corp_300d_bw5.csv"),
        pd.read_csv(f"{outdir}batch2_data_ft_oa_corp_300d_bw5.csv"),
        pd.read_csv(f"{outdir}batch3_data_ft_oa_corp_300d_bw5.csv")
    ])

    # trim the lag off
    df = df[[i for i in df.columns if "lag" not in i]]

    # load and merge the structured data
    strdat = pd.DataFrame(dict(
        month=df.note.apply(lambda x: int(x.split("_")[2][1:])),
        PAT_ID=df.note.apply(lambda x: x.split("_")[3]),
        note=df.note
    ))

    for i in [i for i in os.listdir(outdir) if "_6m" in i]:
        x = pd.read_pickle(f"{outdir}{i}")
        x = x.drop(columns="PAT_ENC_CSN_ID")
        strdat = strdat.merge(x, how='left')
        print(strdat.shape)

    # elixhauser
    elix = pd.read_csv(f"{outdir}elixhauser_scores.csv")
    elix.LATEST_TIME = pd.to_datetime(elix.LATEST_TIME)
    elix['month'] = elix.LATEST_TIME.dt.month + (elix.LATEST_TIME.dt.year - 2018) * 12
    elix = elix.drop(columns=['CSNS', 'LATEST_TIME', 'Unnamed: 0'])
    strdat = strdat.merge(elix, how='left')

    # n_comorb from conc_notes_df
    conc_notes_df = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
    conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (conc_notes_df.LATEST_TIME.dt.year - 2018) * 12
    mm = conc_notes_df[['PAT_ID', 'month', 'n_comorb']]
    strdat = strdat.merge(mm, how='left')

    # add columns for missing values of structured data
    for i in strdat.columns:
        if (strdat[[i]].isna().astype(int).sum() > 0)[0]:
            strdat[[i + "_miss"]] = strdat[[i]].isna().astype(int)
            strdat[[i]] = strdat[[i]].fillna(0)

    # save the column names so that I can find them later
    str_varnames = list(strdat.columns[3:])
    out_varnames = df.columns[5:10]

    # append
    assert (len([i for i in range(len(df.note)) if df.note.iloc[i] != strdat.note.iloc[i]]) == 0)
    df = pd.concat([df.reset_index(drop=True), strdat[str_varnames].reset_index(drop=True)], axis=1)

    # make output dummies
    y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
    df = pd.concat([df, y_dums], axis=1)
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
# embedding_colnames = [i for i in df.columns if re.match("identity|wmean", i)]
input_dims = len(embedding_colnames) + len(str_varnames)
notefiles = os.listdir(embedded_outdir)
month = [notefiles[i].split('_')[2][1:] for i in range(len(notefiles))]
trnotes = [i for i in range(len(notefiles)) if int(month[i]) <=12]
tenotes = [i for i in range(len(notefiles)) if int(month[i]) >12]




def makemodel(nlayers, nfilters, kernel_size, out_kernel_size, batch_normalization, half_dilated, lang = False):
    if half_dilated == True:
        nfilters = nfilters // 2
        drate = int(nlayers) if 2 ** nlayers < 3000 else 10
    inp = Input(shape=(None, input_dims))
    llay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same')(inp)
    if half_dilated == True:
        dlay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same', dilation_rate=2 ** drate)(inp)
        lay = concatenate([llay, dlay], axis=2)
        drate = drate - 1 if drate >= 1 else 1
    lay = LeakyReLU()(lay if half_dilated else llay)
    if batch_normalization == True:
        lay = BatchNormalization()(lay)
    for i in range(nlayers):
        llay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same')(lay)
        if half_dilated == True:
            dlay = Conv1D(filters=nfilters, kernel_size=kernel_size, padding='same', dilation_rate=2 ** drate)(lay)
            lay = concatenate([llay, dlay], axis=2)
            drate = drate - 1 if drate >= 2 else 1
        lay = LeakyReLU()(lay if half_dilated else llay)
        if batch_normalization == True:
            lay = BatchNormalization()(lay)
    if lang == False:
        outlayers = [Conv1D(filters=3, kernel_size=out_kernel_size, activation="softmax", padding='same', name=i)(lay)
                     for i in out_varnames]
    else:
        outlayers = Conv1D(filters = len(embedding_colnames), kernel_size = out_kernel_size, activation = 'linear',
                           padding = 'same')(lay)
    model = Model(inp, outlayers)
    return model

model = makemodel(5, 100, 5, 5, True, True, True)
model.summary()

def note_getter(fi):
    x = np.array(pd.read_pickle(f"{embedded_outdir}{fi}")[embedding_colnames + str_varnames]).astype('float32')
    return np.expand_dims(x,0)

# training function
@tf.function(experimental_relax_shapes=True)
def train(x):
    y = x[:,:,:300] # the outcome doesn't include the structured data
    with tf.GradientTape() as g:
        pred = model(x)
        loss = loss_object(y, pred)
        gradients = g.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


lossvec = []
stopcounter = 0
best = 9999
iter = 0

loss_object = tf.keras.losses.mean_squared_error
optimizer = tf.keras.optimizers.Adam(learning_rate=.00001)

while stopcounter < 500:
    x = note_getter(notefiles[trnotes[np.random.choice(len(trnotes))]])
    loss = np.mean(train(x))
    lossvec.append(loss)
    avgloss = np.mean(lossvec[-20:])
    if avgloss < best:
        best = avgloss
        bestidx = iter
        stopcounter = 0
        best_weights = model.get_weights()
    else:
        stopcounter += 1
    print(f"at {datetime.datetime.now()}")
    print(f"test loss: {avgloss}, "
          f"stopcounter: {stopcounter}, ")
    iter += 1

model.set_weights(best_weights)
model.save_weights(f"{outdir}weights_langmod_v0.h5")

