
'''
AUG 26:  THIS IS A ONE-OFF SCRIPT TO ADD THE BEST LOSS TO THE OUTPUT MODEL PICKLE FILES
PREVIOUSLY, BEFORE PARALLEL, THIS WORKFLOW SAVED HYPERPARAMETERS AND LOSS TO A DATA FRAME.
THIS BROKE IN PARALLEL, AND THE DICT OUTPUT LACKS INFORMATION ON THE BEST LOSS.  THIS PUTS IT IN

'''

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
from tensorflow.keras.layers import concatenate,  \
    LeakyReLU, LSTM, Dropout, Dense, Flatten, Bidirectional
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras import Model, Input, backend
import time
from sklearn.preprocessing import StandardScaler
import copy
from configargparse import ArgParser


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
    options = p.parse_args()
    batchstring = options.batchstring
    mainseed = options.mainseed

    datadir = f"{os.getcwd()}/data/"
    outdir = f"{os.getcwd()}/output/"
    figdir = f"{os.getcwd()}/figures/"
    logdir = f"{os.getcwd()}/logs/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"

    sheepish_mkdir(figdir)
    sheepish_mkdir(logdir)
    sheepish_mkdir(ALdir)
    sheepish_mkdir(f"{ALdir}/ospreds")

    mods_done = [i for i in os.listdir(ALdir) if "model_batch" in i]

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

    # get a vector of non-negatives for case weights
    tr_cw = []
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype \
            ('float32')
        nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
        caseweights = np.ones(df.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[df.note.isin(trnotes)]
        tr_cw.append(tr_caseweights)

    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

    # scaling
    scaler = StandardScaler()
    scaler.fit(df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
    sdf = copy.deepcopy(df)
    sdf[str_varnames + embedding_colnames] = scaler.transform(df[str_varnames + embedding_colnames])

    from pathlib import Path

    for mod in mods_done:
        seed = int(re.split("_|\.", mod)[-2])
        my_file = Path(f"{ALdir}/model_{batchstring}_{seed}.pkl")
        if not my_file.is_file():
            print(mod)
            start = time.time()
            x = read_pickle(f"{ALdir}{mod}")
            model, hps = draw_hps(seed + mainseed)
            assert hps == x['hps']
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

            model.set_weights(x['weights'])

            pred = model.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
            # initialize the loss and the optimizer
            loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
            loss = loss_object(yte, pred)

            hps += (float(loss),)
            hps += (-999,)
            outdict = dict(weights=model.get_weights(),
                           hps=hps)
            write_pickle(outdict, f"{ALdir}/model_{batchstring}_{seed}.pkl")

            tf.keras.backend.clear_session()
            print(f"did {mod} in {time.time()-start}")

