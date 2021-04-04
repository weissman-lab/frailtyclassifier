from utils.misc import read_pickle
import os
import pandas as pd
from utils.organization import find_outdir
from utils.constants import TAGS

outdir = find_outdir()

'''
consolidate single- and multi-task NN performance
'''
def consolidate_NN_perf():
    batches = ['AL01', 'AL02', 'AL03', 'AL04']
    mult_b = []
    single_b = []
    for batch in batches:
        pklpath = f'{outdir}saved_models/{batch}/cv_models/'
        pkls = [i for i in os.listdir(pklpath) if 'model' in i]
        xd = []
        #ID single- and multi-task NN
        multi = [pkl for pkl in pkls if not any(s in pkl for s in TAGS)]
        single = [pkl for pkl in pkls if any(s in pkl for s in TAGS)]
        for pkl in multi:
            x = read_pickle(f"{pklpath}{pkl}")
            d = x['config']
            d = {**d, **x['brier_classwise']}
            d = {**d, **x['brier_aspectwise']}
            d['brier_all'] = x['brier_all']
            d['runtime'] = x['runtime']
            d['ran_when'] = x['ran_when']
            xd.append(d)
        multi = pd.DataFrame(xd)
        multi.shape
        # mean of multi-class scaled briers for all aspects
        multi['brier_mean_aspects'] = multi.loc[:,['Fall_risk', 'Msk_prob', 'Nutrition', 'Resp_imp']].mean(axis = 1)
        # summarize by repeat/fold
        m_col = ['n_dense', 'n_units', 'dropout', 'l1_l2_pen',
           'use_case_weights', 'repeat', 'fold', 'Fall_risk_neg', 'Fall_risk_neut',
           'Fall_risk_pos', 'Msk_prob_neg', 'Msk_prob_neut', 'Msk_prob_pos',
           'Nutrition_neg', 'Nutrition_neut', 'Nutrition_pos', 'Resp_imp_neg',
           'Resp_imp_neut', 'Resp_imp_pos', 'Fall_risk', 'Msk_prob', 'Nutrition',
           'Resp_imp', 'brier_mean_aspects']
        multi_mean_aspect = multi[m_col].groupby(
            ['n_dense', 'n_units', 'dropout', 'l1_l2_pen', 'use_case_weights'],
            as_index=False).mean()
        multi_se_aspect =  multi[m_col].groupby(
            ['n_dense', 'n_units', 'dropout', 'l1_l2_pen', 'use_case_weights'],
            as_index=False).sem()
        multi_agg_aspect = pd.concat([multi_mean_aspect.add_suffix('_mean'),
                                      multi_se_aspect.add_suffix('_se')],
                                     axis=1)
        multi_agg_aspect['batch'] = batch
        mult_b.append(multi_agg_aspect)
        sd = []
        for pkl in single:
            x = read_pickle(f"{pklpath}{pkl}")
            d = x['config']
            d = {**d, **x['brier_classwise']}
            d = {**d, **x['brier_aspectwise']}
            d['brier_all'] = x['brier_all']
            d['runtime'] = x['runtime']
            d['ran_when'] = x['ran_when']
            sd.append(d)
        single = pd.DataFrame(sd)
        # wait to summarize single task NN in R script (will handle single task
        # same as RF & enet)
        single['batch'] = batch
        single_b.append(single)

    #write out
    mtask = pd.concat(mult_b).sort_values(by='brier_mean_aspects_mean',
                                    ascending=False).reset_index()
    mtask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_mtask.csv")

    #write out
    stask = pd.concat(single_b).reset_index()
    stask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_stask.csv")

'''
re-run AL00, AL01, AL02 predictions to generate performance for the historical
learning curve
'''
def rerun_AL00_01_02():
    import os
    from utils.organization import find_outdir
    import pandas as pd
    import re
    from utils.misc import read_pickle
    import numpy as np
    from tensorflow.keras.layers import concatenate, \
        LeakyReLU, LSTM, Dropout, Dense, Flatten, Bidirectional
    from tensorflow.keras.regularizers import l1_l2
    from tensorflow.keras import Model, Input
    from sklearn.preprocessing import StandardScaler
    import copy

    outdir = find_outdir()
    pd.options.display.max_rows = 4000
    pd.options.display.max_columns = 4000

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
            d = Dense(nunits, kernel_regularizer=l1_l2(pen))(
                bid if i == 0 else drp)
            lru = LeakyReLU()(d)
            drp = Dropout(dropout)(lru)
        fl = Flatten()(drp)
        if semipar is True:
            p_inp = Input(shape=(top_shape))
            conc = concatenate([p_inp, fl])
        outlayers = [Dense(3, activation="softmax", name=i,
                           kernel_regularizer=l1_l2(pen))(
            conc if semipar is True else fl)
                     for i in out_varnames]
        if semipar is True:
            model = Model([inp, p_inp], outlayers)
        else:
            model = Model(inp, outlayers)
        return model

    def tensormaker(D, notelist, cols, ws):
        # take a data frame and a list of notes and a list of columns and a window size and return an array for feeding to tensorflow
        note_arrays = [np.array(D.loc[D.note == i, cols]) for i in notelist]
        notelist = []
        for j in range(len(note_arrays)):
            lags, leads = [], []
            for i in range(int(np.ceil(ws / 2)) - 1, 0, -1):
                li = np.concatenate([np.zeros((i, note_arrays[j].shape[1])),
                                     note_arrays[j][:-i]], axis=0)
                lags.append(li)
            assert len(set([i.shape for i in
                            lags])) == 1  # make sure they're all the same size
            for i in range(1, int(np.floor(ws / 2)) + 1, 1):
                li = np.concatenate([note_arrays[j][i:],
                                     np.zeros((i, note_arrays[j].shape[1]))],
                                    axis=0)
                leads.append(li)
            assert len(set([i.shape for i in
                            leads])) == 1  # make sure they're all the same size
            x = np.squeeze(np.stack([lags + [note_arrays[j]] + leads]))
            notelist.append(np.swapaxes(x, 1, 0))
        return np.concatenate(notelist, axis=0)

    def make_y_list(y):
        return [y[:, i * 3:(i + 1) * 3] for i in range(len(out_varnames))]

    # load the notes from 2018
    notes_2018 = sorted(
        [i for i in os.listdir(outdir + "notes_labeled_embedded/") if
         int(i.split("_")[-2][1:]) < 13])

    # drop the notes that aren't in the concatenated notes data frame
    # some notes got labeled and embedded but were later removed from the pipeline
    # on July 14 2020, due to the inclusion of the 12-month ICD lookback
    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    uidstr = ("m" + cndf.month.astype(
        str) + "_" + cndf.PAT_ID + ".csv").tolist()

    notes_2018_in_cndf = [i for i in notes_2018 if
                          "_".join(i.split("_")[-2:]) in uidstr]
    notes_excluded = [i for i in notes_2018 if
                      "_".join(i.split("_")[-2:]) not in uidstr]
    assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)

    # UNSORTED VERSION
    # load the notes from 2018
    UNSORTED_notes_2018 = [i for i in
                           os.listdir(outdir + "notes_labeled_embedded/") if
                           int(i.split("_")[-2][1:]) < 13]
    UNSORTED_notes_2018_in_cndf = [i for i in UNSORTED_notes_2018 if
                                   "_".join(i.split("_")[-2:]) in uidstr]

    df = pd.concat(
        [pd.read_csv(outdir + "notes_labeled_embedded/" + i) for i in
         notes_2018_in_cndf])
    df.drop(columns='Unnamed: 0', inplace=True)

    # define some useful constants
    str_varnames = df.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
    embedding_colnames = [i for i in df.columns if re.match("identity", i)]
    out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
    input_dims = len(embedding_colnames) + len(str_varnames)

    # dummies for the outcomes
    y_dums = pd.concat(
        [pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
    df = pd.concat([y_dums, df], axis=1)

    #########ROUND 0

    mainseed = 8675309

    # split into training and validation
    np.random.seed(mainseed)
    trnotes = np.random.choice(UNSORTED_notes_2018_in_cndf,
                               len(notes_2018_in_cndf) * 2 // 3, replace=False)
    tenotes = [i for i in UNSORTED_notes_2018_in_cndf if i not in trnotes]
    trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
    tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

    r0 = read_pickle('output/saved_models/AL00/model_batch4_92.pkl')
    hps = r0['hps']

    m0 = makemodel(*r0['hps'])
    m0.set_weights(r0['weights'])

    # get a vector of non-negatives for case weights
    tr_cw = []
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if
                           ("_0" not in i) and (v in i)]], axis=1)).astype \
            ('float32')
        nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
        caseweights = np.ones(df.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[df.note.isin(trnotes)]
        tr_cw.append(tr_caseweights)

    # scaling
    scaler = StandardScaler()
    scaler.fit(
        df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
    sdf = copy.deepcopy(df)
    sdf[str_varnames + embedding_colnames] = scaler.transform(
        df[str_varnames + embedding_colnames])

    if hps[-1] is False:  # corresponds with the semipar argument
        Xtr = tensormaker(sdf, trnotes, str_varnames + embedding_colnames,
                          hps[0])
        Xte = tensormaker(sdf, tenotes, str_varnames + embedding_colnames,
                          hps[0])
    else:
        Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])
        Xte_np = tensormaker(sdf, tenotes, embedding_colnames, hps[0])
        Xtr_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
        Xte_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in tenotes])

    ytr = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
    yte = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))

    pred0 = m0.predict([Xte_np, Xte_p] if hps[5] is True else Xte)

    event_rate = y_dums[df.note.isin(tenotes)].mean(axis=0)

    metrics0 = {}  # the metrics dict will have a key for each outcome, and contain objects for computing GOF measures, as well as the GOF measures themselves

    for idx, i in enumerate(out_varnames):
        yhat_er = np.repeat(
            np.expand_dims(np.array(event_rate.filter(like=i)), -1),
            sum(df.note.isin(tenotes)),
            axis=1).T
        yi = np.array(y_dums.loc[
                          df.note.isin(tenotes), [j for j in y_dums.columns if
                                                  i in j]])
        SSE = np.sum((pred0[idx] - yi) ** 2, axis=0)
        SST = np.sum((yhat_er - yi) ** 2, axis=0)
        brier_classwise = 1 - SSE / SST
        brier_aspectwise = 1 - np.mean(SSE) / np.mean(SST)
        metrics0[i] = dict(aspect=i,
                           # event_rate = yhat_er[0],
                           # brier_classwise=brier_classwise,
                           brier_aspectwise=brier_aspectwise,
                           batch='AL00')

    b_asp = []
    for k in metrics0.keys():
        d = pd.DataFrame(metrics0[k], index=[0])
        b_asp.append(d)

    AL00 = pd.concat(b_asp)

    ######## ROUND 1

    mainseed = 29062020

    # split into training and validation
    np.random.seed(mainseed)
    trnotes = np.random.choice(UNSORTED_notes_2018_in_cndf,
                               len(UNSORTED_notes_2018_in_cndf) * 2 // 3,
                               replace=False)
    tenotes = [i for i in UNSORTED_notes_2018_in_cndf if i not in trnotes]
    trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
    tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

    r1 = read_pickle('output/saved_models/AL01/model_batch4_38.pkl')
    hps = r1['hps']
    print(hps)

    m1 = makemodel(*r1['hps'])
    m1.set_weights(r1['weights'])

    # get a vector of non-negatives for case weights
    tr_cw = []
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if
                           ("_0" not in i) and (v in i)]], axis=1)).astype \
            ('float32')
        nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
        caseweights = np.ones(df.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[df.note.isin(trnotes)]
        tr_cw.append(tr_caseweights)

    # scaling
    scaler = StandardScaler()
    scaler.fit(
        df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
    sdf = copy.deepcopy(df)
    sdf[str_varnames + embedding_colnames] = scaler.transform(
        df[str_varnames + embedding_colnames])

    if hps[-1] is False:  # corresponds with the semipar argument
        Xtr = tensormaker(sdf, trnotes, str_varnames + embedding_colnames,
                          hps[0])
        Xte = tensormaker(sdf, tenotes, str_varnames + embedding_colnames,
                          hps[0])
    else:
        Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])
        Xte_np = tensormaker(sdf, tenotes, embedding_colnames, hps[0])
        Xtr_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
        Xte_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in tenotes])

    ytr = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
    yte = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))

    pred1 = m1.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
    event_rate = y_dums[df.note.isin(tenotes)].mean(axis=0)

    metrics1 = {}  # the metrics dict will have a key for each outcome, and contain objects for computing GOF measures, as well as the GOF measures themselves

    for idx, i in enumerate(out_varnames):
        yhat_er = np.repeat(
            np.expand_dims(np.array(event_rate.filter(like=i)), -1),
            sum(df.note.isin(tenotes)),
            axis=1).T
        yi = np.array(y_dums.loc[
                          df.note.isin(tenotes), [j for j in y_dums.columns if
                                                  i in j]])
        SSE = np.sum((pred1[idx] - yi) ** 2, axis=0)
        SST = np.sum((yhat_er - yi) ** 2, axis=0)
        brier_classwise = 1 - SSE / SST
        brier_aspectwise = 1 - np.mean(SSE) / np.mean(SST)
        metrics1[i] = dict(aspect=i,
                           # event_rate = yhat_er[0],
                           # brier_classwise=brier_classwise,
                           brier_aspectwise=brier_aspectwise,
                           batch='AL01')

    b_asp = []
    for k in metrics1.keys():
        d = pd.DataFrame(metrics1[k], index=[0])
        b_asp.append(d)

    AL01 = pd.concat(b_asp)

    ####### ROUND 2

    mainseed = 20200824

    # split into training and validation
    np.random.seed(mainseed)
    trnotes = np.random.choice(notes_2018_in_cndf,
                               len(notes_2018_in_cndf) * 2 // 3, replace=False)
    tenotes = [i for i in notes_2018_in_cndf if i not in trnotes]
    trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
    tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

    r2 = read_pickle('output/saved_models/AL02/model_02_99.pkl')
    hps = r2['hps']
    print(hps)

    m2 = makemodel(*r2['hps'][:-2])
    m2.set_weights(r2['weights'])

    # get a vector of non-negatives for case weights
    tr_cw = []
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if
                           ("_0" not in i) and (v in i)]], axis=1)).astype \
            ('float32')
        nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
        caseweights = np.ones(df.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[df.note.isin(trnotes)]
        tr_cw.append(tr_caseweights)

    # scaling
    scaler = StandardScaler()
    scaler.fit(
        df[str_varnames + embedding_colnames].loc[df.note.isin(trnotes)])
    sdf = copy.deepcopy(df)
    sdf[str_varnames + embedding_colnames] = scaler.transform(
        df[str_varnames + embedding_colnames])

    if hps[-1] is False:  # corresponds with the semipar argument
        Xtr = tensormaker(sdf, trnotes, str_varnames + embedding_colnames,
                          hps[0])
        Xte = tensormaker(sdf, tenotes, str_varnames + embedding_colnames,
                          hps[0])
    else:
        Xtr_np = tensormaker(sdf, trnotes, embedding_colnames, hps[0])
        Xte_np = tensormaker(sdf, tenotes, embedding_colnames, hps[0])
        Xtr_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in trnotes])
        Xte_p = np.vstack(
            [sdf.loc[sdf.note == i, str_varnames] for i in tenotes])

    ytr = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
    yte = make_y_list(np.vstack(
        [sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))

    pred2 = m2.predict([Xte_np, Xte_p] if hps[5] is True else Xte)
    event_rate = y_dums[df.note.isin(tenotes)].mean(axis=0)

    metrics2 = {}  # the metrics dict will have a key for each outcome, and contain objects for computing GOF measures, as well as the GOF measures themselves

    for idx, i in enumerate(out_varnames):
        yhat_er = np.repeat(
            np.expand_dims(np.array(event_rate.filter(like=i)), -1),
            sum(df.note.isin(tenotes)),
            axis=1).T
        yi = np.array(y_dums.loc[
                          df.note.isin(tenotes), [j for j in y_dums.columns if
                                                  i in j]])
        SSE = np.sum((pred2[idx] - yi) ** 2, axis=0)
        SST = np.sum((yhat_er - yi) ** 2, axis=0)
        brier_classwise = 1 - SSE / SST
        brier_aspectwise = 1 - np.mean(SSE) / np.mean(SST)
        metrics2[i] = dict(aspect=i,
                           # event_rate = yhat_er[0],
                           # brier_classwise=brier_classwise,
                           brier_aspectwise=brier_aspectwise,
                           batch='AL02')

    b_asp = []
    for k in metrics2.keys():
        d = pd.DataFrame(metrics2[k], index=[0])
        b_asp.append(d)

    AL02 = pd.concat(b_asp)

    AL_00_01_02 = pd.concat([AL00, AL01, AL02]) \
        [['brier_aspectwise', 'batch']].groupby('batch').mean()

    print(AL_00_01_02)

    AL_00_01_02.to_csv(f"{outdir}figures_tables/AL_00_01_02.csv")

'''
Combine historical performance from AL00/AL01/AL02 with current performance 
from AL03 and beyond
'''
def AL_learning_curve(latest_batch):
    historic = pd.read_csv(
        f"{outdir}figures_tables/AL_00_01_02.csv")

    current = pd.read_csv(
        f"{outdir}saved_models/{latest_batch}/learning_curve_mtask.csv")

    current = current[~current.batch.isin(['AL01', 'AL02'])]

    current = current.loc[current.reset_index().groupby(['batch'])\
        ['brier_mean_aspects_mean'].idxmax()] \
        [['batch', 'brier_mean_aspects_mean', 'brier_mean_aspects_se']]

    current['brier_aspectwise'] = current['brier_mean_aspects_mean']
    current['brier_aspectwise_se'] = current['brier_mean_aspects_se']

    historic['brier_aspectwise_se'] = 0

    cols = ['batch', 'brier_aspectwise', 'brier_aspectwise_se']

    learning_curve_AL = pd.concat([current[cols], historic[cols]])

    learning_curve_AL.to_csv(f"{outdir}figures_tables/learning_curve_AL.csv")


if __name__ == "__main__":
    main()