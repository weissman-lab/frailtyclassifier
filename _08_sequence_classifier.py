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
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import re
import numpy as np
from _99_project_module import nrow, write_txt, ncol
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, concatenate, Flatten, Conv1D, \
    LSTM, LeakyReLU, BatchNormalization
from tensorflow.keras import Model, Input, backend
import copy

outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"

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

# add columns for missing values of structured data
for i in strdat.columns:
    if any(strdat[[i]].isna()):
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
df.to_pickle(f"{outdir}moddat.pkl")



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
input_dims = 600 + len(str_varnames)
out_varnames = df.columns[5:10]
df['month'] = df.note.apply(lambda x: int(x.split("_")[2][1:]))
trnotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) <= 12]
tenotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
note_lengths = df.note.value_counts()
# embedding_colnames = [i for i in df.columns if re.match("identity", i)]
embedding_colnames = [i for i in df.columns if re.match("identity|wmean", i)]

# simple bilstm with parametric part for the structured data
# inp = Input(shape=(None, input_dims - len(str_varnames)))
# str_input = Input((len(str_varnames)))
# LSTM_forward = LSTM(50, return_sequences=True)(inp)
# LSTM_backward = LSTM(50, return_sequences=True, go_backwards=True)(inp)
# LSTM_backward = backend.reverse(LSTM_backward, axes=1)
# conc = concatenate([LSTM_forward, LSTM_backward, str_input], axis=2)
# outlayers = [Dense(3, activation="softmax", name=i)(conc) for i in out_varnames]
# model = Model([inp, str_input], outlayers)
# model.summary()

inp = Input(shape=(None, input_dims))
str_input = Input((len(str_varnames)))
LSTM_forward = LSTM(50, return_sequences=True)(inp)
LSTM_backward = LSTM(50, return_sequences=True, go_backwards=True)(inp)
LSTM_backward = backend.reverse(LSTM_backward, axes=1)
conc = concatenate([LSTM_forward, LSTM_backward], axis=2)
outlayers = [Dense(3, activation="softmax", name=i)(conc) for i in out_varnames]
model = Model(inp, outlayers)
model.summary()


# initialize the bias terms with the logits of the proportions
def logit(x):
    return 1 / (1 + np.exp(-x))


def inv_logit(x):
    return np.log(x/(1-x))

w = model.get_weights()

for i in range(5):
    props = np.array([inv_logit(np.mean(df[out_varnames[i]]==-1)),
                      inv_logit(np.mean(df[out_varnames[i]]==0)),
                      inv_logit(np.mean(df[out_varnames[i]]==1))])
    print(props)
    pos = 9-i*2
    print(pos)
    print(w[-pos].shape)
    w[-pos] = w[-pos]*0+props

model.set_weights(w)

def cutter_padder(df, note, cols):
    di = df.loc[df.note == note, cols]
    x = np.array(di).astype('float32')
    return x



# batchmaker function
def batchmaker(note):
    '''
    function to make batches to feed to TF.  Because some notes are longer 
    than others, and to prevent oversampling the middles of notes, this 
    function will randomly cut the note and pad with zeros, even in cases 
    where the note length is greater then the number of inpout timesteps.
    '''
    
    emdat = cutter_padder(df, note, embedding_colnames)
    strdat = cutter_padder(df, note, str_varnames)
    X = np.concatenate([emdat, strdat], axis = 1)
    # y variables
    yvars = cutter_padder(df, note, y_dums.columns)
    # break out the outcomes into a list of tensors
    y_list_of_tensors = [tf.convert_to_tensor(yvars[:, i * 3:(i + 1) * 3]) for i in range(5)]
    output = dict(x=tf.convert_to_tensor(np.expand_dims(X,0)),
                  y=y_list_of_tensors)
    return output


loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
optimizer = tf.keras.optimizers.Adam(learning_rate=.0001)


# training function
@tf.function
def train(x, y):
    with tf.GradientTape() as g:
        pred = model(x)
        loss = loss_object(y, pred)
        gradients = g.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def makeplot():
    fig, ax = plt.subplots(figsize=(16, 10), ncols=2, nrows=1, sharex=False)
    ax[0].plot(list(range(len(oslossvec))), oslossvec, label='test rmse')
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('iteration // test_rate')
    ax[0].grid('on')
    for i in range(5):
        ax[1].plot(list(range(len(oslossvec))), osrmse[:iter//test_rate+1,i], label=out_varnames[i])
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('iteration // test_rate')
    ax[1].grid('on')
    ax[1].legend()
    fig.savefig(f"{figdir}LSTM_mar26.png")
    plt.close('all')

# training loop
osrmse = np.empty((1000,5))
oslossvec = []
test_rate = 5
stopcounter = 0
best = 9999
iter = 0

y_preds = copy.deepcopy(y_dums)
y_preds['note'] = df.note
y_preds.replace([-1,0,1], [np.nan, np.nan, np.nan], inplace = True)
y_dums.to_csv(f"{outdir}y_dums.csv")

while stopcounter < len(tenotes):
    if iter % test_rate == 0:
        rnote = np.random.choice(len(tenotes))
        tebatch = batchmaker(tenotes[rnote])
        pred = model(tebatch['x'])
        osloss = loss_object(tebatch['y'], pred)
        oslossvec.append(osloss)
        rmse = tf.keras.losses.mean_squared_error(tebatch['y'], pred)**.5
        osrmse[iter//test_rate, :] = np.mean(rmse, axis=(1, 2))
        # weighted averages
        w_avg_osloss = np.mean(oslossvec[-10:])
        # update predictions data frame
        for i in range(5):
            y_preds.loc[y_preds.note == tenotes[rnote],
                        [out_varnames[i]+"_"+j for j in ["-1", "0", "1"]]] = np.squeeze(pred[i])
            y_preds.to_csv(f"{outdir}iterpreds.csv")
        if w_avg_osloss < best:
            best = w_avg_osloss
            bestidx = iter
            stopcounter = 0
            model.save_weights(f"{outdir}weights_LSTM_mar26.h5")
        else:
            stopcounter += 1
        print(f"at {datetime.datetime.now()}")
        print(f"test loss: {osloss}, "
              f"test_rmse: {osrmse[iter//test_rate, :]}, "
              f"stopcounter: {stopcounter}, "
              f"iter: {iter}")
        makeplot()
    batch = batchmaker(trnotes[np.random.choice(len(trnotes))])
    train(batch['x'], batch['y'])
    iter+=1
    print(iter)


tf.keras.backend.clear_session()
