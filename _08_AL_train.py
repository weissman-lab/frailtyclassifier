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
from utils.prefit import make_model, make_transformers_model
from utils.figures_tables import lossplot
from utils.constants import TAGS, SENTENCE_LENGTH
import numpy as np
import tensorflow as tf

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


class Trainer:
    def __init__(self, batchstring, task, dev=False, model_type='w2v', earlystopping=False):
        assert task in ['multi', 'Resp_imp', 'Msk_prob', 'Nutrition', 'Fall_risk']
        self.outdir = f"./output/"
        self.datadir = f"./data/"
        self.ALdir = f"{self.outdir}saved_models/AL{batchstring}/"
        sheepish_mkdir(f"{self.ALdir}figures")
        self.model_type = model_type
        if self.model_type is None:
            self.model_type = 'w2v'
        self.task = task
        self.batchstring = batchstring
        self.earlystopping = earlystopping
        self.dev = dev
        # defined in methods:
        self.cfg = None
        self.bestmods = None
        self.hdict = None
        self.cvmodpath = None
        self.tr_pids = None

    def get_config(self):
        cvmodpath = f"{self.ALdir}cv_models/"
        if self.model_type != "w2v":
            cvmodpath += self.model_type + "/"
        self.cvmodpath = cvmodpath
        cvmods = os.listdir(cvmodpath)
        if self.task == 'multi':
            pkls = [i for i in cvmods if any(re.findall("\d\.pkl", i))]
        else:
            pkls = [i for i in cvmods if any(re.findall(self.task, i))]
        cvdf = []
        for pkl in pkls:
            if (self.dev == True) | (self.batchstring == "01"):
                try:
                    x = read_pickle(f"{cvmodpath}/{pkl}")
                    d = x['config']
                    d['brier_all'] = x['brier_all']
                    d['fn'] = pkl
                    cvdf.append(d)
                except:
                    pass
            else:
                x = read_pickle(f"{cvmodpath}/{pkl}")
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
        self.bestmods = bestmods
        self.cfg = read_pickle(f"{cvmodpath}/{bestmods[0]}")['config']

    def lossplot(self):
        hdict = dict(L=[],
                     fL=[],
                     mL=[],
                     nL=[],
                     rL=[])
        for b in self.bestmods:
            x = read_pickle(f"{self.cvmodpath}/{b}")
            hdict['L'].append(x['history']['val_loss'])
            if self.task == 'multi':
                hdict['fL'].append(x['history']['val_Fall_risk_loss'])
                hdict['mL'].append(x['history']['val_Msk_prob_loss'])
                hdict['nL'].append(x['history']['val_Nutrition_loss'])
                hdict['rL'].append(x['history']['val_Resp_imp_loss'])

        self.hdict = hdict
        medianlen = np.median([len(i) for i in hdict['L']]) // 1
        self.cfg['medianlen'] = medianlen
        if (self.task == "multi") & (self.model_type == 'w2v'):
            ff = lossplot(hdict, medianlen)
            suffix = "" if self.task == "multi" else f"_{self.task}"
            ff.savefig(f"{self.ALdir}figures/cvlossplot{suffix}.pdf")
            ff.savefig(f"{self.ALdir}figures/cvlossplot{suffix}.png", dpi=400)

    def fit(self):
        ##################
        # load data
        if self.earlystopping == False:
            df = pd.read_csv(f"{self.ALdir}processed_data/full_set/full_df.csv", index_col=0)
            sent = df['sentence']
        else:
            df_tr = pd.read_csv(f"{self.ALdir}processed_data/full_set_earlystopping/rNone_fNone_tr_df.csv", index_col=0)
            df_va = pd.read_csv(f"{self.ALdir}processed_data/full_set_earlystopping/rNone_fNone_va_df.csv", index_col=0)
            sent_tr = df_tr['sentence']
            sent_va = df_va['sentence']
            str_varnames = [i for i in df_tr.columns if re.match("pca[0-9]", i)]

        ###################
        # create model and vectorizer
        mmfun = make_model if self.model_type == 'w2v' else make_transformers_model
        if self.earlystopping == False:
            emb_filename = f"embeddings_{self.model_type}_final.npy"  # only used for transformers
        else:
            emb_filename = f"embeddings_{self.model_type}_final_tr.npy"  # only used for transformers

        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model, vectorizer = mmfun(emb_path=f"{self.datadir}w2v_oa_all_300d.bin",
                                      sentence_length=SENTENCE_LENGTH,
                                      meta_shape=len(str_varnames),
                                      tags=[self.task] if self.task != 'multi' else TAGS,
                                      train_sent=sent if self.earlystopping == False else sent_tr,
                                      test_sent=None if self.earlystopping == False else sent_va,
                                      l1_l2_pen=self.cfg['l1_l2_pen'],
                                      n_units=self.cfg['n_units'],
                                      n_dense=self.cfg['n_dense'],
                                      dropout=self.cfg['dropout'],
                                      ALdir=self.ALdir,
                                      embeddings=self.model_type,
                                      emb_filename=emb_filename
                                      )

            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             patience=25,
                                                             restore_best_weights=True)

            model.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4))

        #####################
        # prepare data for tensorfow
        if self.model_type == 'w2v':
            if self.earlystopping == False:
                text = vectorizer(np.array([[s] for s in sent]))
                test_nan_inf(text)
            else:
                text_tr = vectorizer(np.array([[s] for s in sent_tr]))
                test_nan_inf(text_tr)
                text_va = vectorizer(np.array([[s] for s in sent_va]))
                test_nan_inf(text_va)

        if self.earlystopping == False:
            labels = []
            if self.task == 'multi':
                for n in TAGS:
                    lab = tf.convert_to_tensor(df[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype='float32')
                    labels.append(lab)
                assert all([all(tf.reduce_mean(tf.cast(i, dtype='float32'), axis=0) % 1 > 0) for i in labels])
            else:
                labels = tf.convert_to_tensor(df[[f"{self.task}_neg",
                                                  f"{self.task}_neut",
                                                  f"{self.task}_pos"]], dtype='float32')
                assert all(tf.reduce_mean(tf.cast(labels, dtype='float32'), axis=0) % 1 > 0)
            struc = tf.convert_to_tensor(df[str_varnames], dtype='float32')
            test_nan_inf(struc)
            test_nan_inf(labels)

        else:
            labels_tr = []
            labels_va = []
            if self.task == 'multi':
                for n in TAGS:
                    lab = tf.convert_to_tensor(df_tr[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype='float32')
                    labels_tr.append(lab)
                    lab = tf.convert_to_tensor(df_va[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype='float32')
                    labels_va.append(lab)
                assert all([all(tf.reduce_mean(tf.cast(i, dtype='float32'), axis=0) % 1 > 0) for i in labels_tr])
                assert all([all(tf.reduce_mean(tf.cast(i, dtype='float32'), axis=0) % 1 > 0) for i in labels_va])
            else:
                labels_tr = tf.convert_to_tensor(df_tr[[f"{self.task}_neg",
                                                        f"{self.task}_neut",
                                                        f"{self.task}_pos"]], dtype='float32')
                labels_va = tf.convert_to_tensor(df_va[[f"{self.task}_neg",
                                                        f"{self.task}_neut",
                                                        f"{self.task}_pos"]], dtype='float32')
                assert all(tf.reduce_mean(tf.cast(labels_tr, dtype='float32'), axis=0) % 1 > 0)
                assert all(tf.reduce_mean(tf.cast(labels_va, dtype='float32'), axis=0) % 1 > 0)

            for v in str_varnames:  # lose outliers in PCA
                df_tr.loc[df_tr[v] > 4, v] = 4
                df_tr.loc[df_tr[v] < -4, v] = -4
                df_va.loc[df_va[v] > 4, v] = 4
                df_va.loc[df_va[v] < -4, v] = -4

            struc_tr = tf.convert_to_tensor(df_tr[str_varnames], dtype='float32')
            struc_va = tf.convert_to_tensor(df_va[str_varnames], dtype='float32')

            test_nan_inf(struc_tr)
            test_nan_inf(labels_tr)
            test_nan_inf(struc_va)
            test_nan_inf(labels_va)

        #############################
        # initialize the bias terms with the logits of the proportions
        w = model.get_weights()
        # set the bias terms to the proportions
        if self.task == 'multi':
            for i, yi in enumerate(labels if self.earlystopping == False else labels_tr):
                props = inv_logit(tf.reduce_mean(yi, axis=0).numpy())
                pos = 7 - i * 2
                w[-pos] = w[-pos] * 0 + props
        else:
            props = inv_logit(tf.reduce_mean(labels if self.earlystopping == False else labels_tr, axis=0).numpy())
            pos = 1
            w[-pos] = w[-pos] * 0 + props
        model.set_weights(w)

        if self.earlystopping == False:
            X = [vectorizer['tr'], struc] if not self.model_type == 'w2v' else [text, struc]

            history = model.fit(x=X,
                                y=labels,
                                epochs=int(self.cfg['medianlen']) if self.dev == False else 1,
                                batch_size=32,
                                verbose=1)
        else:
            Xtr = [vectorizer['tr'], struc_tr] if not self.model_type == 'w2v' else [text_tr, struc_tr]
            Xva = [vectorizer['va'], struc_va] if not self.model_type == 'w2v' else [text_va, struc_va]
            history = model.fit(x=Xtr,
                                y=labels_tr,
                                validation_data=(Xva, labels_va),
                                callbacks=earlystopping,
                                epochs=int(self.cfg['medianlen']) if self.dev == False else 1,
                                batch_size=32,
                                verbose=1)

        #
        if self.task == 'multi' & self.earlystopping == False:
            final_hdict = dict(L=history.history['loss'],
                               fL=history.history['Fall_risk_loss'],
                               mL=history.history['Msk_prob_loss'],
                               nL=history.history['Nutrition_loss'],
                               rL=history.history['Resp_imp_loss'])
            fff = lossplot(self.hdict, self.cfg['medianlen'], final_hdict)
            fff.savefig(f"{self.ALdir}figures/cvlossplot_w_final_{self.model_type}.pdf")
            fff.savefig(f"{self.ALdir}figures/cvlossplot_w_final_{self.model_type}.png", dpi=400)

        # collect the output
        outdict = dict(config=self.cfg,
                       history=history.history,
                       weights=model.get_weights(),
                       ran_when=datetime.datetime.now(),
                       ran_on=tf.config.list_physical_devices()
                       )
        if self.dev == False:
            sheepish_mkdir(f"{self.ALdir}/final_model")
            suffix = "" if self.task == "multi" else f"_{self.task}"
            suffix += f"{'_' + self.model_type if self.model_type is not 'w2v' else ''}"
            suffix += f"{'_earlystopping' if self.earlystopping == True else ''}"
            write_pickle(outdict, f"{self.ALdir}/final_model/model_final_{self.batchstring}{suffix}.pkl")
        else:
            print('it seems to work!')

    def run(self):
        self.get_config()
        self.lossplot()
        self.fit()


def main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
    p.add("--singletask", action='store_true')
    p.add("--dev", action='store_true')
    p.add("--model_type")
    p.add("--earlystopping", action='store_true')
    options = p.parse_args()
    batchstring = options.batchstring
    singletask = options.singletask
    model_type = options.model_type
    earlystopping = options.earlystopping

    dev = options.dev
    if singletask == False:
        trobj = Trainer(batchstring=batchstring, task='multi', dev=dev, model_type=model_type,
                        earlystopping=earlystopping)
        trobj.run()
    else:
        for task in TAGS:
            print(f"starting {task}")
            trobj = Trainer(batchstring=batchstring, task=task, dev=dev, model_type=model_type,
                            earlystopping=earlystopping)
            trobj.run()


if __name__ == '__main__':
    main()
    # self = Trainer(batchstring='01', task='multi', dev=True, model_type='bioclinicalbert',
    #                earlystopping=True)
    # self.run()


def old_main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
    p.add("--singletask", action='store_true')
    options = p.parse_args()
    batchstring = options.batchstring
    singletask = options.singletask

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
