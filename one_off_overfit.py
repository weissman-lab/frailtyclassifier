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
    print("this is hijacked for the one-off script")
    hps = (int(np.random.choice(list(range(4, 40)))),  # window size
           int(np.random.choice(list(range(1, 10)))),  # n dense
           int(2 ** np.random.choice(list(range(5, 11)))),  # n units
           float(np.random.uniform(low=0, high=.5)),  # dropout
           float(10 ** np.random.uniform(-8, -2)),  # l1/l2 penalty
           bool(np.random.choice(list(range(2)))))  # semipar
    hps = (3, 1, 2**1, 0.0, 0.0, True)
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
    # batchstring = "oneoff"
    # ##########################################
    p = ArgParser()
    p.add("--batchstring", help="the batch number", type=str)
    p.add("--mainseed", help="path to the embeddings file", type=int)
    # p.add("--init", help="initialize_inprog?", action="store_true")
    options = p.parse_args()
    batchstring = options.batchstring
    mainseed = options.mainseed
    # initialize_inprog = options.init

    datadir = f"{os.getcwd()}/data/"
    outdir = f"{os.getcwd()}/output/"
    figdir = f"{os.getcwd()}/figures/"
    logdir = f"{os.getcwd()}/logs/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"
    assert batchstring not in ['00', '01', '02']

    sheepish_mkdir(figdir)
    sheepish_mkdir(logdir)
    sheepish_mkdir(ALdir)
    sheepish_mkdir(f"{ALdir}/ospreds")


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


    seed = 0
    try:
        np.random.seed(mainseed + seed)
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model, hps = draw_hps(seed + mainseed)

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
            print(seed)
            print(hps)

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

        # checkpoint the models so that they can restart mid-stride
        # make a directory for each model's checkpoints
        # check for the presence of a checkpoint, and load it if it exists
        checkpoint_filepath = f"{ALdir}ckpt{seed}/checkpoint"
        sheepish_mkdir("/" + "/".join(checkpoint_filepath.split("/")[:-1]))
        # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        #     filepath=checkpoint_filepath,
        #     save_weights_only=True,
        #     monitor='val_loss',
        #     save_best_only=True)

        try:
            model.load_weights(checkpoint_filepath)
            print('loaded weights from previous go-round')
        except:
            print("didn't find any previous weights.  here's the contents of the checkpoint filepath:")
            print(os.listdir("/" + "/".join(checkpoint_filepath.split("/")[:-1])))
            pass


        # earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
        #                                                           patience=20,
        #                                                           restore_best_weights=True)
        log_dir = outdir + "/logs/fit/seed_" + str(seed) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # sheepish_mkdir(log_dir)
        # pd.DataFrame({"seed": int(i)}, index=[i]).to_csv(f"{log_dir}/job{i}")

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit([Xtr_np, Xtr_p] if hps[5] is True else Xtr, ytr,
                  batch_size=32,
                  epochs=10000,
                  callbacks=[tensorboard_callback],
                  sample_weight=tr_cw,
                  verbose=1,
                  validation_data=([Xte_np, Xte_p], yte) if hps[5] is True else (Xte, yte))
        pred = model.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
        # initialize the loss and the optimizer
        loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        loss = loss_object(yte, pred)

        hps += (float(loss),)
        hps += (time.time() - start_time,)
        outdict = dict(weights=model.get_weights(),
                       hps=hps)
        write_pickle(outdict, f"{ALdir}/model_{batchstring}_{seed}.pkl")

        catprop = np.mean([np.mean(x[:, 1]) for x in pred])

        print(f"at {datetime.datetime.now()}")
        print(f"test loss: {loss}")
        print("quantiles of the common category")
        for i in range(4):
            print(np.quantile([pred[i][:, 1]], [.1, .2, .3, .4, .5, .6, .7, .8, .9]))

        tf.keras.backend.clear_session()

    except Exception as e:
        # put the borken job back on the shelf
        pd.DataFrame({"seed": seed}, index=[seed]).to_csv(f"{ALdir}TBD/job{seed}")
        send_message_to_slack(e)
        print(e)
        logf = open(f"{logdir}seed{seed}.log", "w")
        logf.write(str(e))
        logf.close()
    n_remaining = len(os.listdir(f"{ALdir}/TBD/"))

    # """
    # Now figure out the winner and ingest the unlabeled notes
    # """
    #
    # print('starting entropy search')
    #
    # hpdf = pd.read_json(f"{ALdir}hpdf.json")
    # winner = hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]
    #
    # # load it
    # best_model = pd.read_pickle(f"{ALdir}model_batch4_{int(winner.idx)}.pkl")
    # model = makemodel(*best_model['hps'])
    # model.set_weights(best_model['weights'])
    #
    # # find all the notes to check
    # notefiles = [i for i in os.listdir(f"{outdir}embedded_notes/")]
    # # lose the ones that are in the trnotes:
    # trstubs = ["_".join(i.split("_")[-2:]) for i in trnotes]
    # testubs = ["_".join(i.split("_")[-2:]) for i in tenotes]
    # notefiles = [i for i in notefiles if (i not in trstubs) and (i not in testubs) and ("DS_Store" not in i)]
    # # and lose the ones that aren't 2018
    # notefiles = [i for i in notefiles if int(i.split("_")[2][1:]) <= 12]
    # # lose the ones that aren't in the cndf
    # # the cndf was cut on July 14, 2020 to only include notes from PTs with qualifying ICD codes from the 12 months previous
    # cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    # cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    # cndf['month'] = cndf.LATEST_TIME.dt.month + (
    #         cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    # cndf_notes = ("embedded_note_m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".pkl").tolist()
    # notefiles = list(set(notefiles) & set(cndf_notes))
    #
    #
    # def h(x):
    #     """entropy"""
    #     return -np.sum(x * np.log(x), axis=1)
    #
    #
    # # now loop through them, normalize, and predict
    # if "entropies_of_unlableled_notes.pkl" not in os.listdir(ALdir):
    #     edicts = []
    #     N = 0
    #     for i in notefiles:
    #         if f"pred{i}.pkl" not in os.listdir(f"{ALdir}ospreds/"):
    #             r = get_entropy_stats(i)
    #             write_pickle(r, f"{ALdir}ospreds/pred{i}.pkl")
    #         else:
    #             r = read_pickle(f"{ALdir}ospreds/pred{i}.pkl")
    #         r.pop("pred")
    #         print(r)
    #         edicts.append(r)
    #         print(i)
    #         N += 1
    #         print(N)
    #
    #     # res = pd.concat([res, pd.DataFrame([i for i in edicts if i is not None])])
    #     res = pd.DataFrame([i for i in edicts if i is not None])
    #     res.to_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")
    # else:
    #     res = pd.read_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")
    #
    # colnames = res.columns[1:].tolist()
    # fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    # for i in range(5):
    #     for j in range(5):
    #         if i == j:
    #             ax[i, j].hist(res[colnames[i]])
    #             ax[i, j].set_xlabel(colnames[i])
    #         elif i > j:
    #             ax[i, j].scatter(res[colnames[i]], [res[colnames[j]]], s=.5)
    #             ax[i, j].set_xlabel(colnames[i])
    #             ax[i, j].set_ylabel(colnames[j])
    # plt.tight_layout()
    # fig.savefig(f"{ALdir}entropy_summaries.pdf")
    # plt.show()
    #
    # # pull the best notes
    # cndf['note'] = "embedded_note_m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".pkl"
    # res = res.merge(cndf[['note', 'combined_notes']])
    #
    # best = res.sort_values("hmean_above_average", ascending=False).head(30)
    # best['PAT_ID'] = best.note.apply(lambda x: x.split("_")[3][:-4])
    # best['month'] = best.note.apply(lambda x: x.split("_")[2][1:])
    # cndf['month'] = cndf.LATEST_TIME.dt.month
    #
    # selected_notes = []
    # for i in range(len(best)):
    #     ni = cndf.combined_notes.loc[(cndf.month == int(best.month.iloc[i])) & (cndf.PAT_ID == best.PAT_ID.iloc[i])]
    #     assert len(ni) == 1
    #     selected_notes.append(ni)
    #
    # for i, n in enumerate(selected_notes):
    #     # assert 5==0, 'SET THIS BACK TO 25 NEXT TIME YOU RUN IT'
    #     fn = f"AL{batchstring}_v2{'ALTERNATE' if i > 24 else ''}_m{best.month.iloc[i]}_{best.PAT_ID.iloc[i]}.txt"
    #     print(fn)
    #     write_txt(n.iloc[0], f"{ALdir}{fn}")
    #
    # # post-analysis
    # hpdf.to_csv(f"{ALdir}hpdf_for_R.csv")
    #
    # # get the files
    # try:
    #     os.mkdir(f"{ALdir}best_notes_embedded")
    # except Exception:
    #     pass
    # if 'crandrew' in os.getcwd():
    #     assert 5 == 0  # if you ver need to make this work again, fix it to not rely on grace
    #     for note in best.note:
    #         cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/saved_models/AL{batchstring}/ospreds/pred{note}.pkl" \
    #               f" {ALdir}ospreds/"
    #         os.system(cmd)
    #     for note in best.note:
    #         cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/embedded_notes/{note}" \
    #               f" {ALdir}best_notes_embedded/"
    #         os.system(cmd)
    # elif 'hipaa_garywlab' in os.getcwd():
    #     for note in best.note:
    #         cmd = f"cp {ALdir}/ospreds/pred{note}.pkl" \
    #               f" {ALdir}best_notes_embedded/"
    #         os.system(cmd)
    #     for note in best.note:
    #         cmd = f"cp /project/hipaa_garywlab/frailty/output/embedded_notes/{note}" \
    #               f" {ALdir}best_notes_embedded/"
    #         os.system(cmd)
    #
    # predfiles = os.listdir(f"{ALdir}best_notes_embedded")
    # predfiles = [i for i in predfiles if "predembedded" in i]
    # enotes = os.listdir(f"{ALdir}best_notes_embedded")
    # enotes = [i for i in enotes if "predembedded" not in i]
    #
    # j = 0
    # for j, k in enumerate(predfiles):
    #     p = read_pickle(f"{ALdir}best_notes_embedded/{predfiles[j]}")
    #     ID = re.sub('.pkl', '', "_".join(predfiles[j].split("_")[2:]))
    #     emat = np.stack([h(x) for x in p['pred']]).T
    #     emb_note = read_pickle(f"{ALdir}best_notes_embedded/{[x for x in enotes if ID in x][0]}")
    #     fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
    #     for i in range(4):
    #         ax[i].plot(p['pred'][i][:, 0], label='neg')
    #         ax[i].plot(p['pred'][i][:, 2], label='pos')
    #         hx = h(p['pred'][i])
    #         ax[i].plot(hx + 1, label='entropy')
    #         ax[i].legend()
    #         ax[i].axhline(1)
    #         ax[i].set_ylabel(out_varnames[i])
    #         ax[i].set_ylim(0, 2.1)
    #         maxH = np.argmax(emat[:, i])
    #         span = emb_note.token.iloc[(maxH - best_model['hps'][0] // 2):(maxH + best_model['hps'][0] // 2)]
    #         string = " ".join(span.tolist())
    #         ax[i].text(maxH, 2.1, string, horizontalalignment='center')
    #     plt.tight_layout()
    #     plt.show()
    #     fig.savefig(f"{ALdir}best_notes_embedded/predplot_w_best_span_{enotes[j]}.pdf")