import os
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
if 'crandrew' in os.getcwd():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from _99_project_module import inv_logit, send_message_to_slack, write_pickle, read_pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import concatenate, \
    LeakyReLU, LSTM, Dropout, Dense, Flatten, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Model, Input, backend
import time
from sklearn.preprocessing import StandardScaler
import copy
from configargparse import ArgParser
from pathlib import Path


def sheepish_mkdir(path):
    import os
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def makemodel(window_size, n_dense, nunits,
              dropout, pen, semipar):
    if semipar is True:
        base_shape = input_dims - len(str_varnames)
        top_shape = input_dims - len(embedding_colnames)
    else:
        base_shape = input_dims
    inp = Input(shape=(window_size, base_shape))
    LSTM_layer = LSTM(nunits, return_sequences=True,
                      kernel_regularizer=l1_l2(pen))
    bid = Bidirectional(LSTM_layer)(inp)
    # dense
    for i in range(n_dense):
        d = Dense(nunits, kernel_regularizer=l1_l2(pen))(bid if i == 0 else drp)
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


def get_entropy_stats(i, return_raw=False):
    try:
        start = time.time()
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
        end = time.time()

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
                   pred=pred,
                   # time
                   time=end - start
                   )
        return out
    except Exception as e:
        print(e)
        print(i)


if __name__ == '__main__':

    # #########################################
    # # set some globals
    # batchstring = "02"
    # # set the seed and define the training and test sets
    # # mainseed = 8675309
    # # mainseed= 29062020 # 29 June 2020
    # mainseed = 20200813  # 13 August 2020 batch 2
    # mainseed = 20200824  # 24 August 2020 batch 2 reboot, after fixing sortedness issue
    # initialize_inprog = True
    # ##########################################
    p = ArgParser()
    p.add("--batchstring", help="the batch number", type=str)
    p.add("--mainseed", help="path to the embeddings file", type=int)
    p.add("--init", help="form the hpdf? only needs to be done once", action="store_true")
    options = p.parse_args()
    batchstring = options.batchstring
    assert batchstring is not None
    mainseed = options.mainseed
    form_hpdf = options.init


    datadir = f"{os.getcwd()}/data/"
    outdir = f"{os.getcwd()}/output/"
    figdir = f"{os.getcwd()}/figures/"
    logdir = f"{os.getcwd()}/logs/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"

    sheepish_mkdir(figdir)
    sheepish_mkdir(logdir)
    sheepish_mkdir(ALdir)
    sheepish_mkdir(f"{ALdir}/ospreds")

    # assemble the hpdf.
    if form_hpdf:
        hpdf = []
        for i in range(100):
            try:
                x = read_pickle(f"{ALdir}model_{batchstring}_{i}.pkl")
                y = (i,) + x['hps']
                hpdf.append(y)
            except:
                print(f"problem at {i}")
                pass

        hpdf = pd.DataFrame(hpdf, columns=['seed',
                                           'window_size',
                                           'n_dense',
                                           'n_units',
                                           'dropout',
                                           'l1_l2',
                                           'semipar',
                                           'best_loss',
                                           'time_to_convergence',
                                           'hand_loss'])

        assert len(hpdf) == 100, f"only {len(hpdf)} model files loaded!"
        hpdf.to_json(f"{ALdir}hpdf.json")
        hpdf.to_csv(f"{ALdir}hpdf.csv")
    else:
        hpdf = pd.read_json(f"{ALdir}hpdf.json")

    # load the notes from 2018
    notes_2018 = sorted([i for i in os.listdir(outdir + "notes_labeled_embedded/") if int(i.split("_")[-2][1:]) < 13])

    # drop the notes that aren't in the concatenated notes data frame
    # some notes got labeled and embedded but were later removed from the pipeline
    # on July 14 2020, due to the inclusion of the 12-month ICD lookback
    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()

    notes_2018_in_cndf = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) in uidstr]
    notes_excluded = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) not in uidstr]
    assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)

    # write_txt(",".join(["_".join(i.split("_")[-2:]) for i in notes_excluded]), f"{outdir}cull_list_15jul.txt")

    df = pd.concat([pd.read_csv(outdir + "notes_labeled_embedded/" + i) for i in notes_2018])
    df.drop(columns='Unnamed: 0', inplace=True)

    # split into training and validation
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

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # scaling
    scaler = StandardScaler()
    scaler.fit(df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
    sdf = copy.deepcopy(df)
    sdf[str_varnames + embedding_colnames] = scaler.transform(df[str_varnames + embedding_colnames])




    print('starting entropy search')

    hpdf = pd.read_json(f"{ALdir}hpdf.json")
    winner = hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]

    # load it
    best_model = read_pickle(f"{ALdir}model_{batchstring}_{i}.pkl")

    model = makemodel(*best_model['hps'][:-2])
    model.set_weights(best_model['weights'])

    # find all the notes to check
    notefiles = [i for i in os.listdir(f"{outdir}embedded_notes/")]
    # lose the ones that are in the trnotes:
    trstubs = ["_".join(i.split("_")[-2:]) for i in trnotes]
    testubs = ["_".join(i.split("_")[-2:]) for i in tenotes]
    notefiles = [i for i in notefiles if (i not in trstubs) and (i not in testubs) and ("DS_Store" not in i)]
    # and lose the ones that aren't 2018
    notefiles = [i for i in notefiles if int(i.split("_")[2][1:]) <= 12]
    # lose the ones that aren't in the cndf
    # the cndf was cut on July 14, 2020 to only include notes from PTs with qualifying ICD codes from the 12 months previous
    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    cndf_notes = ("embedded_note_m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".pkl").tolist()
    notefiles = list(set(notefiles) & set(cndf_notes))


    def h(x):
        """entropy"""
        return -np.sum(x * np.log(x), axis=1)


    # randomly sort notefiles
    np.random.seed(int(time.time()))
    notefiles = list(np.random.choice(notefiles, len(notefiles), replace = False))

    N=0
    for i in notefiles:
        my_file = Path(f"{ALdir}ospreds/pred{i}.pkl")
        if not my_file.is_file():
            r = get_entropy_stats(i)
            write_pickle(r, f"{ALdir}ospreds/pred{i}.pkl")
            r.pop("pred")
            print(r)
            print(i)
            N += 1
            print(N)


