'''
From the CV experiments, identify the best hyperparameters and the optimal numbers of epochs
Then get the model into training.
Arguments should just be the batchstring
'''
from configargparse import ArgParser
import os
import datetime
import re
import pandas as pd
from utils.misc import (read_pickle, write_pickle, sheepish_mkdir,
                        inv_logit, test_nan_inf)
from utils.prefit import make_model
from utils.figures_tables import lossplot
from utils.constants import TAGS, SENTENCE_LENGTH
import numpy as np
import tensorflow as tf


# pd.options.display.max_rows = 4000
# pd.options.display.max_columns = 4000


def main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
    options = p.parse_args()
    batchstring = options.batchstring

    outdir = f"{os.getcwd()}/output/"
    datadir = f"{os.getcwd()}/data/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"
    sheepish_mkdir(f"{ALdir}figures")

    ########################################
    # load the pickles and figure out which configuration is best
    cvmods = os.listdir(f"{ALdir}cv_models/")
    pkls = [i for i in cvmods if any(re.findall("\d\.pkl", i))]
    cvdf = []
    for pkl in pkls:
        print(pkl)
        x = read_pickle(f"{ALdir}cv_models/{pkl}")
        d = x['config']
        d['brier_all'] = x['brier_all']
        d['fn'] = pkl
        cvdf.append(d)

    df = pd.DataFrame(cvdf)
    dfa = df.groupby(['n_dense', 'n_units', 'dropout', 'l1_l2_pen'])['brier_all'].mean().reset_index()
    dfb = df.groupby(['n_dense', 'n_units', 'dropout', 'l1_l2_pen'])['brier_all'].std().reset_index()
    dfa = dfa.rename(columns={'brier_all': "mu"})
    dfb = dfb.rename(columns={'brier_all': "sig"})
    dfagg = dfa.merge(dfb)
    dfagg = dfagg.reset_index(drop=False)
    dfagg = dfagg.sort_values('mu')

    bestmods = df.merge(dfagg.loc[dfagg.mu == dfagg.mu.max()])

    bestmods = bestmods.fn.tolist()
    bestmods.sort()

    ###########################
    # generate the loss plot
    hdict = dict(L=[],
                 fL=[],
                 mL=[],
                 nL=[],
                 rL=[])
    for b in bestmods:
        x = read_pickle(f"{ALdir}cv_models/{b}")
        hdict['L'].append(x['history']['val_loss'])
        hdict['fL'].append(x['history']['val_Fall_risk_loss'])
        hdict['mL'].append(x['history']['val_Msk_prob_loss'])
        hdict['nL'].append(x['history']['val_Nutrition_loss'])
        hdict['rL'].append(x['history']['val_Resp_imp_loss'])

    medianlen = np.median([len(i) for i in hdict['L']]) // 1
    ff = lossplot(hdict, medianlen)
    ff.savefig(f"{ALdir}figures/cvlossplot.pdf")
    ff.savefig(f"{ALdir}figures/cvlossplot.png", dpi=400)

    ##################
    # load data
    df = pd.read_csv(f"{ALdir}processed_data/full_set/full_df.csv", index_col=0)
    sent = df['sentence']
    str_varnames = [i for i in df.columns if re.match("pca[0-9]", i)]

    ###################
    # create model and vectorizer
    cfg = read_pickle(f"{ALdir}cv_models/{bestmods[0]}")['config']
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model, vectorizer = make_model(emb_path=f"{datadir}w2v_oa_all_300d.bin",
                                       sentence_length=SENTENCE_LENGTH,
                                       meta_shape=len(str_varnames),
                                       tags=TAGS,
                                       train_sent=sent,
                                       l1_l2_pen=cfg['l1_l2_pen'],
                                       n_units=cfg['n_units'],
                                       n_dense=cfg['n_dense'],
                                       dropout=cfg['dropout'])

        model.compile(loss='categorical_crossentropy',
                      optimizer=tf.keras.optimizers.Adam(1e-4))

    #####################
    # prepare data for tensorfow
    text = vectorizer(np.array([[s] for s in sent]))

    labels = []
    for n in TAGS:
        lab = tf.convert_to_tensor(df[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype='float32')
        labels.append(lab)

    struc = tf.convert_to_tensor(df[str_varnames], dtype='float32')

    test_nan_inf(text)
    test_nan_inf(labels)
    test_nan_inf(struc)
    # test for constant columns in labels
    assert all([all(tf.reduce_mean(tf.cast(i, dtype='float32'), axis=0) % 1 > 0) for i in labels])

    #############################
    # initialize the bias terms with the logits of the proportions
    w = model.get_weights()
    # set the bias terms to the proportions
    for i, yi in enumerate(labels):
        props = inv_logit(tf.reduce_mean(yi, axis=0).numpy())
        pos = 7 - i * 2
        w[-pos] = w[-pos] * 0 + props
    model.set_weights(w)

    history = model.fit(x=[text, struc],
                        y=labels,
                        epochs=int(medianlen),
                        batch_size=32,
                        verbose=1)

    #
    final_hdict = dict(L=history.history['loss'],
                       fL=history.history['Fall_risk_loss'],
                       mL=history.history['Msk_prob_loss'],
                       nL=history.history['Nutrition_loss'],
                       rL=history.history['Resp_imp_loss'])
    fff = lossplot(hdict, medianlen, final_hdict)
    fff.savefig(f"{ALdir}figures/cvlossplot_w_final.pdf")
    fff.savefig(f"{ALdir}figures/cvlossplot_w_final.png", dpi=400)

    # collect the output
    outdict = dict(config=cfg,
                   history=history.history,
                   weights=model.get_weights(),
                   ran_when=datetime.datetime.now(),
                   ran_on=tf.config.list_physical_devices()
                   )
    sheepish_mkdir(f"{ALdir}/final_model")
    write_pickle(outdict, f"{ALdir}/final_model/model_final_{batchstring}.pkl")


if __name__ == '__main__':
    main()
