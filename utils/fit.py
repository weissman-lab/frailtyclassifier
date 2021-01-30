
import os
import re
import time
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score, average_precision_score
from utils.prefit import make_model
import datetime
from utils.misc import write_pickle, sheepish_mkdir, test_nan_inf


def AL_CV(index = 0,
          batchstring = '03',
          n_dense = 5,
          n_units = 64,
          dropout = .3,
          l1_l2_pen = 1e-4,
          use_case_weights = False,
          repeat = 1,
          fold = 0):

    #################
    SENTENCE_LENGTH = 20 # set standard sentence length. Inputs will be truncated or padded
    TAGS = ['Fall_risk', 'Msk_prob',  'Nutrition', 'Resp_imp']
    outdir = f"{os.getcwd()}/output/"
    datadir = f"{os.getcwd()}/data/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"
    cv_savepath = f"{ALdir}cv_models"
    sheepish_mkdir(cv_savepath)
    savename = f"{cv_savepath}/model_pickle_cv_{index}.pkl"
    if savename in os.listdir(cv_savepath):
        return 
    else:
        ##################
        # load data
        df_tr = pd.read_csv(f"{ALdir}processed_data/trvadata/r{repeat}_f{fold}_tr_df.csv", index_col = 0)
        df_va = pd.read_csv(f"{ALdir}processed_data/trvadata/r{repeat}_f{fold}_va_df.csv", index_col = 0)
        case_weights = pd.read_csv(f"{ALdir}processed_data/caseweights/r{repeat}_f{fold}_tr_caseweights.csv", index_col = 0)
        
        train_sent = df_tr['sentence']
        test_sent = df_va['sentence']
        
        str_varnames = [i for i in df_tr.columns if re.match("pca[0-9]",i)]
        
        
        ###################
        # create model and vectorizer
        mirrored_strategy = tf.distribute.MirroredStrategy()

        with mirrored_strategy.scope():
            model, vectorizer = make_model(emb_path = f"{datadir}w2v_oa_all_300d.bin",
                               sentence_length = SENTENCE_LENGTH,
                               meta_shape = len(str_varnames),
                               tags = TAGS,
                               train_sent = train_sent,
                               l1_l2_pen = l1_l2_pen,
                               n_units = n_units,
                               n_dense = n_dense,
                               dropout = dropout)
            
            earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                             patience=25,
                                                             restore_best_weights=True)
            model.compile(loss='categorical_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(1e-4))
        
        #####################
        # prepare data for tensorfow
        tr_text = vectorizer(np.array([[s] for s in train_sent]))
        va_text = vectorizer(np.array([[s] for s in test_sent]))
        
        tr_labels = []
        va_labels = []
        for n in TAGS:
            tr = tf.convert_to_tensor(df_tr[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype = 'float32')
            va = tf.convert_to_tensor(df_va[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype = 'float32')
            tr_labels.append(tr)
            va_labels.append(va)
        
        tr_struc = tf.convert_to_tensor(df_tr[str_varnames], dtype = 'float32')
        va_struc = tf.convert_to_tensor(df_va[str_varnames], dtype = 'float32')
        
        case_weights_tensor_list = [tf.convert_to_tensor(case_weights[t+"_cw"]) for t in TAGS]
        
        test_nan_inf(tr_text)
        test_nan_inf(va_text)
        test_nan_inf(tr_labels)
        test_nan_inf(va_labels)
        test_nan_inf(tr_struc)
        test_nan_inf(va_struc)
        # test for constant columns in labels
        assert all([all(tf.reduce_mean(tf.cast(i, dtype = 'float32'), axis = 0) % 1 > 0) for i in tr_labels])
        assert all([all(tf.reduce_mean(tf.cast(i, dtype = 'float32'), axis = 0) % 1 > 0) for i in va_labels])
        
        #############################
        # fit the model

        
        start_time = time.time()
        history = model.fit(x = [tr_text, tr_struc],
                            y = tr_labels,
                            validation_data=(
                                  [va_text, va_struc], va_labels),
                          epochs=1000,
                          batch_size=32,
                          verbose = 1,
                          sample_weight=case_weights_tensor_list if use_case_weights == True else None,
                          callbacks=earlystopping)
        runtime = time.time()-start_time
        ################################
        # predictions and metrics
        va_preds = model.predict([va_text, va_struc])
        
        subtags = [f"{t}_{i}" for t in TAGS for i in ['neg', 'neut', 'pos']]
        
        va_preds = tf.concat(va_preds, axis=1).numpy()
        va_y = tf.concat(va_labels, axis=1).numpy()
        
        event_rate = np.stack([va_y.mean(axis = 0) for i in range(va_y.shape[0])])
        SSE = np.sum((va_preds - va_y)**2, axis = 0)
        SST = np.sum((event_rate - va_y)**2, axis = 0)
        brier_classwise = 1 - SSE/SST
        brier_aspectwise = [1-np.mean(SSE[(i*3):((i+1)*3)])/np.mean(SST[(i*3):((i+1)*3)]) for i in range(4)]
        brier_all = 1-np.sum(SSE)/np.sum(SST)
        
        brier_classwise = {k: brier_classwise[i] for i, k in enumerate(subtags)}
        brier_aspectwise = {k: brier_aspectwise[i] for i, k in enumerate(TAGS)}

        [i for i in enumerate(subtags)]
        auroc = [{t:roc_auc_score(va_y[:,i], va_preds[:,i])} for i, t in enumerate(subtags)]
        auprc = [{t:average_precision_score(va_y[:,i], va_preds[:,i])} for i, t in enumerate(subtags)]
        
        
        va_preds_df = pd.DataFrame(va_preds, columns = subtags)
        va_preds_df.insert(0, 'sentence_id', df_va.sentence_id)
        va_label_df = pd.DataFrame(va_y, columns = subtags)
        va_label_df.insert(0, 'sentence_id', df_va.sentence_id)
        va_eventrate_df = pd.DataFrame(event_rate, columns = subtags)
        va_eventrate_df.insert(0, 'sentence_id', df_va.sentence_id)
        # collect the output
        config_dict = dict(batchstring = batchstring,
                                     n_dense = n_dense,
                                     n_units = n_units,
                                     dropout = dropout,
                                     l1_l2_pen = l1_l2_pen,
                                     use_case_weights = use_case_weights,
                                     repeat = repeat,
                                     fold = fold)
        outdict = dict(config = config_dict,
                       history = history.history,
                       va_preds = va_preds_df,
                       va_label = va_label_df,
                       va_eventrate = va_eventrate_df,
                       brier_classwise = brier_classwise,
                       brier_aspectwise = brier_aspectwise,
                       brier_all = brier_all,
                       auroc = auroc,
                       auprc = auprc,
                       weights = model.get_weights(),
                       cohort = dict(tr = list(df_tr.PAT_ID.unique()),
                                     va = list(df_va.PAT_ID.unique())),
                       runtime = runtime,
                       ran_when = datetime.datetime.now(),
                       ran_on = tf.config.list_physical_devices()
                       )
        report = '*********************************************\n\n'
        report += f"Config: \n{config_dict}\n"
        report += f"Brier classwise: \n{brier_classwise}\n"
        report += f"Brier aspectwise: \n{brier_aspectwise}\n"
        report += f"Brier all: \n{brier_all}\n"
        report += f"AUROC: \n{auroc}\n"
        report += f"AUPRC: \n{auprc}\n"
        print(report)

        
        write_pickle(outdict, savename)
        return 0


if __name__ == "__main__":
    pass