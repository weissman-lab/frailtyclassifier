import os
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
if 'crandrew' in os.getcwd():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from _99_project_module import inv_logit, send_message_to_slack, write_pickle, read_pickle
from _99_project_module import write_txt
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


#########################################
# set some globals
batchstring = "01"
# set the seed and define the training and test sets
#mainseed = 8675309
mainseed= 29062020 # 29 June 2020
##########################################


datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"
ALdir = f"{outdir}saved_models/AL{batchstring}/"


try:
    os.mkdir(ALdir)
except Exception:
    pass

try:
    os.mkdir(f"{ALdir}ospreds")
except Exception:
    pass

# load the notes from 2018
notes_2018 = [i for i in os.listdir(outdir + "notes_labeled_embedded/") if int(i.split("_")[-2][1:]) < 13]
df = pd.concat([pd.read_csv(outdir + "notes_labeled_embedded/" + i) for i in notes_2018])
df.drop(columns='Unnamed: 0', inplace=True)


np.random.seed(mainseed) 
trnotes = np.random.choice(notes_2018, len(notes_2018) * 2 // 3, replace=False)
tenotes = [i for i in notes_2018 if i not in trnotes]
trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

# define some useful constants
str_varnames = df.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# dummies for the outcomes
y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
df = pd.concat([y_dums, df], axis=1)

# get a vector of non-negatives for case weights
tr_cw = []
for v in out_varnames:
    non_neutral = np.array(np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype \
        ('float32')
    nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
    caseweights = np.ones(df.shape[0])
    caseweights[non_neutral.astype(bool)] *= nnweight
    tr_caseweights = caseweights[df.note.isin(trnotes)]
    tr_cw.append(tr_caseweights)


def makemodel(window_size, n_dense, nunits,
              dropout, pen, semipar):
    if semipar is True:
        base_shape = input_dims - len(str_varnames)
        top_shape = input_dims - len(embedding_colnames)
    else:
        base_shape = input_dims
    inp = Input(shape=(window_size, base_shape))
    LSTM_forward = LSTM(nunits, return_sequences=True,
                        kernel_regularizer=l1_l2(pen))(inp)
    LSTM_backward = LSTM(nunits, return_sequences=True, go_backwards=True,
                         kernel_regularizer=l1_l2(pen))(inp)
    LSTM_backward = backend.reverse(LSTM_backward, axes=1)
    conc = concatenate([LSTM_forward, LSTM_backward], axis=2)
    # dense
    for i in range(n_dense):
        d = Dense(nunits, kernel_regularizer=l1_l2(pen))(conc if i == 0 else drp)
        lru = LeakyReLU()(d)
        drp = Dropout(dropout)(lru)
    fl = Flatten()(drp)
    if semipar is True:
        p_inp = Input(shape=(top_shape))
        conc = concatenate([p_inp, fl])
    outlayers = [Dense(3, activation="softmax", name=i,
                       kernel_regularizer=l1_l2(pen))(conc if semipar is True else fl)
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
           int(2 ** np.random.choice(list(range(5, 11)))),  # n units
           float(np.random.uniform(low=0, high=.5)),  # dropout
           float(10 ** np.random.uniform(-8, -2)),  # l1/l2 penalty
           bool(np.random.choice(list(range(2)))))  # semipar
    model = makemodel(*hps)
    return model, hps

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

# initialize a df for results
hpdf = pd.DataFrame(dict(idx=list(range(100)),
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
        for i in range(int(np.ceil(ws / 2)) - 1, 0, -1):
            li = np.concatenate([np.zeros((i, note_arrays[j].shape[1])), note_arrays[j][:-i]], axis=0)
            lags.append(li)
        assert len(set([i.shape for i in lags])) == 1  # make sure they're all the same size
        for i in range(1, int(np.floor(ws / 2)) + 1, 1):
            li = np.concatenate([note_arrays[j][i:], np.zeros((i, note_arrays[j].shape[1]))], axis=0)
            leads.append(li)
        assert len(set([i.shape for i in leads])) == 1  # make sure they're all the same size
        x = np.squeeze(np.stack([lags + [note_arrays[j]] + leads]))
        notelist.append(np.swapaxes(x, 1, 0))
    return np.concatenate(notelist, axis=0)


def make_y_list(y):
    return [y[:, i * 3:(i + 1) * 3] for i in range(len(out_varnames))]


# scaling
scaler = StandardScaler()
scaler.fit(df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
sdf = copy.deepcopy(df)
sdf[str_varnames + embedding_colnames] = scaler.transform(df[str_varnames + embedding_colnames])

# look for hpdf
try:
    hpdf = pd.read_json(f"{ALdir}hpdf.json")
    # hpdf.drop(columns = 'oob', inplace = True)
    startpos = hpdf.dropna().idx.max() + 1
except Exception:
    startpos = 0


for seed in range(startpos, 100):
    try:
        np.random.seed(mainseed + seed)
        model, hps = draw_hps(seed + mainseed)
        for i in range(1, 7):  # put the hyperparameters in the hpdf
            hpdf.loc[seed, hpdf.columns[i]] = hps[i - 1]

        # put the data in arrays for modeling, expanding out to the window size
        # only converting the test into tensors, to facilitate indexing
        if hps[-1] is False:  # corresponds with the semipar argument
            Xtr = tensormaker(sdf, trnotes, str_varnames + embedding_colnames, hps[0])
            Xte = tensormaker(sdf, tenotes, str_varnames + embedding_colnames, hps[0])
        else:
            Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])
            Xte_np = tensormaker(sdf, tenotes, embedding_colnames, hps[0])
            Xtr_p = np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
            Xte_p = np.vstack([sdf.loc[sdf.note == i, str_varnames] for i in tenotes])
        ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
        yte = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))

        print("\n\n********************************\n\n")
        print(hpdf.iloc[seed])

        start_time = time.time()

        # initialize the bias terms with the logits of the proportions
        w = model.get_weights()
        # set the bias terms to the proportions
        for i in range(4):
            props = np.array([inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == -1)),
                              inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 0)),
                              inv_logit(np.mean(df.loc[df.note.isin(trnotes), out_varnames[i]] == 1))])
            pos = 7 - i * 2
            w[-pos] = w[-pos] * 0 + props
        model.set_weights(w)

        model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                      loss={'Msk_prob': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                            'Nutrition': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                            'Resp_imp': tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                            'Fall_risk': tf.keras.losses.CategoricalCrossentropy(from_logits=False)})

        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=20,
                                                    restore_best_weights=True)
        model.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
                  batch_size=256,
                  epochs=1000,
                  callbacks=[callback],
                  sample_weight=tr_cw,
                  verbose=0,
                  validation_data=([Xte_np, Xte_p], yte) if hps[5] is True else (Xte, yte))
        outdict = dict(weights=model.get_weights(),
                       hps=hps)
        write_pickle(outdict, f"{ALdir}/model_batch4_{seed}.pkl")

        pred = model.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
        # initialize the loss and the optimizer
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = loss_object(yte, pred)

        catprop = np.mean([np.mean(x[:, 1]) for x in pred])

        print(f"at {datetime.datetime.now()}")
        print(f"test loss: {loss}")
        print("quantiles of the common category")
        for i in range(4):
            print(np.quantile([pred[i][:, 1]], [.1, .2, .3, .4, .5, .6, .7, .8, .9]))

        tf.keras.backend.clear_session()
        hpdf.loc[seed, 'best_loss'] = float(loss)
        hpdf.loc[seed, 'time_to_convergence'] = time.time() - start_time
        hpdf.to_json(f"{ALdir}hpdf.json")
    except Exception as e:
        send_message_to_slack(e)
        break

"""
Now figure out the winner and ingest the unlabeled notes
"""

print('starting entropy search')

hpdf = pd.read_json(f"{ALdir}hpdf.json")
winner = hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]


# load it
best_model = pd.read_pickle(f"{ALdir}model_batch4_{int(winner.idx)}.pkl")
model = makemodel(*best_model['hps'])
model.set_weights(best_model['weights'])

# find all the notes to check
notefiles = [i for i in os.listdir(f"{outdir}embedded_notes/")]
# lose the ones that are in the trnotes:
trstubs = ["_".join(i.split("_")[-2:]) for i in trnotes]
testubs = ["_".join(i.split("_")[-2:]) for i in tenotes]

notefiles = [i for i in notefiles if (i not in trstubs) and (i not in testubs) and ("DS_Store" not in i)]
# and lose the ones that aren't 2018
notefiles = [i for i in notefiles if int(i.split("_")[2][1:]) <= 12]

def h(x):
    """entropy"""
    return -np.sum(x * np.log(x), axis=1)


# now loop through them, normalize, and predict
def get_entropy_stats(i, return_raw=False):
    try:
        note = pd.read_pickle(f"{outdir}embedded_notes/{i}")
        note[str_varnames + embedding_colnames] = scaler.transform(note[str_varnames + embedding_colnames])
        note['note'] = "foo"
        if best_model['hps'][-1] is False:  # corresponds with the semipar argument
            Xte = tensormaker(note, ['foo'], str_varnames + embedding_colnames, best_model['hps'][0])
        else:
            Xte_np = tensormaker(note, ['foo'], embedding_colnames, best_model['hps'][0])
            Xte_p = np.vstack([note[str_varnames] for i in ['foo']])

        pred = model.predict([Xte_np, Xte_p] if best_model['hps'][5] is True else Xte)
        hmat = np.stack([h(i) for i in pred])

        out = dict(note=i,
                   hmean=np.mean(hmat),
                   # compute average entropy, throwing out lower half
                   hmean_top_half=np.mean(hmat[hmat > np.median(hmat)]),
                   # compute average entropy, throwing out those that are below the (skewed) average
                   hmean_above_average=np.mean(hmat[hmat > np.mean(hmat)]),
                   # maximum
                   hmax=np.max(hmat),
                   # top decile average
                   hdec=np.mean(hmat[hmat > np.quantile(hmat, .9)]),
                   # the raw predictions
                   pred=pred
                   )
        return out
    except Exception as e:
        print(e)
        print(i)

if "entropies_of_unlableled_notes.pkl" not in os.listdir(ALdir):
    edicts = []
    N = 0
    for i in notefiles:
        if f"pred{i}.pkl" not in os.listdir(f"{ALdir}ospreds/"):
            r = get_entropy_stats(i)
            write_pickle(r, f"{ALdir}ospreds/pred{i}.pkl")
        else:
            r = read_pickle(f"{ALdir}ospreds/pred{i}.pkl")
        r.pop("pred")
        print(r)
        edicts.append(r)
        print(i)
        N += 1
        print(N)
    
    # res = pd.concat([res, pd.DataFrame([i for i in edicts if i is not None])])
    res = pd.DataFrame([i for i in edicts if i is not None])
    res.to_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")
else:
    res = pd.read_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")

colnames = res.columns[1:].tolist()
fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        if i == j:
            ax[i, j].hist(res[colnames[i]])
            ax[i, j].set_xlabel(colnames[i])
        elif i > j:
            ax[i, j].scatter(res[colnames[i]], [res[colnames[j]]], s=.5)
            ax[i, j].set_xlabel(colnames[i])
            ax[i, j].set_ylabel(colnames[j])
plt.tight_layout()
fig.savefig(f"{ALdir}entropy_summaries.pdf")
plt.show()

# pull the best notes
cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]

best = res.sort_values("hmean_above_average").tail(25)
best['PAT_ID'] = best.note.apply(lambda x: x.split("_")[3][:-4])
best['month'] = best.note.apply(lambda x: x.split("_")[2][1:])
cndf['month'] = cndf.LATEST_TIME.dt.month

selected_notes = []
for i in range(len(best)):
    ni = cndf.combined_notes.loc[(cndf.month == int(best.month.iloc[i])) & (cndf.PAT_ID == best.PAT_ID.iloc[i])]
    assert len(ni) == 1
    selected_notes.append(ni)

for i, n in enumerate(selected_notes):
    fn = f"AL{batchstring}_m{best.month.iloc[i]}_{best.PAT_ID.iloc[i]}.txt"
    write_txt(n.iloc[0], f"{ALdir}{fn}")


# post-analysis
hpdf.to_csv(f"{ALdir}hpdf_for_R.csv")

# get the files
try:
    os.mkdir(f"{ALdir}best_notes_embedded")
except Exception:
    pass
if 'crandrew' in os.getcwd():
    assert 5==0 # if you ver need to make this work again, fix it to not rely on grace
    for note in best.note:
        cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/saved_models/AL{batchstring}/ospreds/pred{note}.pkl" \
              f" {ALdir}ospreds/"
        os.system(cmd)
    for note in best.note:
        cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/embedded_notes/{note}" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)
elif 'hipaa_garywlab' in os.getcwd():
    for note in best.note:
        cmd = f"cp {ALdir}/ospreds/pred{note}.pkl" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)
    for note in best.note:
        cmd = f"cp /project/hipaa_garywlab/frailty/output/embedded_notes/{note}" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)


predfiles = os.listdir(f"{ALdir}best_notes_embedded")
predfiles = [i for i in predfiles if "predembedded" in i]
enotes = os.listdir(f"{ALdir}best_notes_embedded")
enotes = [i for i in enotes if "predembedded" not in i]

j = 0
for j, k in enumerate(predfiles):
    p = read_pickle(f"{ALdir}best_notes_embedded/{predfiles[j]}")
    ID = re.sub('.pkl', '', "_".join(predfiles[j].split("_")[2:]))
    emat = np.stack([h(x) for x in p['pred']]).T
    emb_note = read_pickle(f"{ALdir}best_notes_embedded/{[x for x in enotes if ID in x][0]}")
    fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
    for i in range(4):
        ax[i].plot(p['pred'][i][:,0], label='neg')
        ax[i].plot(p['pred'][i][:,2], label='pos')
        hx = h(p['pred'][i])
        ax[i].plot(hx+1, label='entropy')
        ax[i].legend()
        ax[i].axhline(1)
        ax[i].set_ylabel(out_varnames[i])
        ax[i].set_ylim(0, 2.1)
        maxH = np.argmax(emat[:,i])
        span = emb_note.token.iloc[(maxH - best_model['hps'][0]//2):(maxH + best_model['hps'][0]//2)]
        string = " ".join(span.tolist())
        ax[i].text(maxH, 2.1, string, horizontalalignment='center')
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{ALdir}best_notes_embedded/predplot_w_best_span_{enotes[j]}.pdf")

