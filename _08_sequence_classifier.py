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
out_varnames = df.columns[5:10]
df['month'] = df.note.apply(lambda x: int(x.split("_")[2][1:]))
trnotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) <= 12]
tenotes = [i for i in df.note.unique() if int(i.split("_")[2][1:]) > 12]
note_lengths = df.note.value_counts()


# nlayers = 10
# nfilters = 100
# kernel_size = 10
# out_kernel_size = 3
# batch_normalization = True
# half_dilated = True


def makemodel(nlayers, nfilters, kernel_size, out_kernel_size, batch_normalization, half_dilated):
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
           bool(np.random.choice(list(range(2)))),  # batch normalization
           bool(np.random.choice(list(range(2)))))  # half-dilation
    model = makemodel(*hps)
    return model, hps


def cutter_padder(df, note, cols):
    di = df.loc[df.note == note, cols]
    x = np.array(di).astype('float32')
    return x


# batchmaker function
def batchmaker(note, return_y_as_list=True):
    emdat = cutter_padder(df, note, embedding_colnames)
    strdat = cutter_padder(df, note, str_varnames)
    X = np.concatenate([emdat, strdat], axis=1)
    # y variables
    yvars = cutter_padder(df, note, y_dums.columns)
    if return_y_as_list == True:
        # break out the outcomes into a list of tensors
        y_list_of_tensors = [tf.convert_to_tensor(yvars[:, i * 3:(i + 1) * 3]) for i in range(5)]
        output = dict(x=tf.convert_to_tensor(np.expand_dims(X, 0)),
                      y=y_list_of_tensors)
    else:
        output = dict(x=tf.convert_to_tensor(np.expand_dims(X, 0)),
                      y=yvars)
    return output


def makeplot():
    fig, ax = plt.subplots(figsize=(24, 9), ncols=3, nrows=1, sharex=False)
    ax[0].plot(list(range(len(oslossvec))), oslossvec, label='test rmse')
    ax[0].plot(list(range(len(oslossvec))), [np.mean(oslossvec[:i][-20:]) for i in len(oslossvec)])
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('iteration // test_rate')
    ax[0].grid('on')
    ax[0].axhline(y=best, color='r', linestyle='-')
    for i in range(5):
        ax[1].plot(list(range(len(oslossvec))), osrmse[:iter // test_rate + 1, i], label=out_varnames[i])
    ax[1].set_ylabel('RMSE')
    ax[1].set_xlabel('iteration // test_rate')
    ax[1].grid('on')
    ax[1].legend()
    for i in range(5):
        ax[2].plot(list(range(len(oslossvec))), rmse_rare[:iter // test_rate + 1, i], label=out_varnames[i])
    ax[2].set_ylabel('RMSE, non-neutral')
    ax[2].set_xlabel('iteration // test_rate')
    ax[2].grid('on')
    ax[2].legend()
    fig.savefig(f"{figdir}LSTM_mar26.png")
    plt.close('all')


def get_rare_rmse(y, pred):
    rmsevec = []
    for i in range(len(y)):
        idx = np.where(y[i][:, 1] == 0)
        if len(idx[0]) == 0:
            rmsevec.append(np.nan)
        else:
            br = y[i].numpy()[idx[0], :]
            pr = np.squeeze(pred[i])[idx[0], :]
            rmsevec.append(np.mean((br - pr) ** 2) ** .5)
    return rmsevec




# initialize a df for results
hpdf = pd.DataFrame(dict(idx=list(range(1000)),
                         nlayers=np.nan,
                         nfilters=np.nan,
                         kernel_size=np.nan,
                         out_kernel_size=np.nan,
                         batch_normalization=np.nan,
                         half_dilated=np.nan,
                         time_to_convergence=np.nan,
                         best_loss=np.nan))

model_iteration = 0
for seed in range(1000):
    try:
        np.random.seed(seed)

        model, hps = draw_hps(seed)
        for i in range(1, 7):
            hpdf.loc[model_iteration, hpdf.columns[i]] = hps[i - 1]

        print("\n\n********************************\n\n")
        print(hpdf.iloc[model_iteration])

        start_time = time.time()

        # initialize the bias terms with the logits of the proportions
        w = model.get_weights()

        for i in range(5):
            props = np.array([inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == -1)),
                              inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == 0)),
                              inv_logit(np.mean(df.loc[df.month <= 12, out_varnames[i]] == 1))])
            # print(props)
            pos = 9 - i * 2
            # print(pos)
            # print(w[-pos].shape)
            w[-pos] = w[-pos] * 0 + props

        model.set_weights(w)

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


        def test():
            lossvec = []
            wtvec = []
            for i in tenotes:
                tebatch = batchmaker(i)
                pred = model(tebatch['x'])
                lossvec.append(loss_object(tebatch['y'], pred))
                wtvec.append(note_lengths[i])
            avg_loss = sum([float(lossvec[i]) * wtvec[i] for i in range(len(lossvec))]) / sum(wtvec)
            return avg_loss


        # training loop
        osrmse = np.empty((10000, 5))
        oslossvec = []
        test_rate = 75
        stopcounter = 0
        best = 9999
        iter = 0

        while stopcounter < 5:
            if iter % test_rate == 0:
                osloss = test()
                oslossvec.append(osloss)
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
            batch = batchmaker(trnotes[np.random.choice(len(trnotes))])
            train(batch['x'], batch['y'])
            iter += 1

        tf.keras.backend.clear_session()
        hpdf.loc[model_iteration, 'best_loss'] = best
        hpdf.loc[model_iteration, 'time_to_convergence'] = time.time() - start_time
        hpdf.to_csv(f"{outdir}hyperparameter_gridsearch_results.csv")
        model_iteration += 1
    except Exception as e:
        send_message_to_slack(e)
        break



