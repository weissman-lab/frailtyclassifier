
'''
From the CV experiments, identify the best hyperparameters and the optimal numbers of epochs
Then get the model into training.
Arguments should just be the batchstring
'''

import os
import re
import pandas as pd
from utils.misc import read_pickle, write_pickle, sheepish_mkdir
from utils.prefit import make_model
from utils.figures_tables import lossplot
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

##################
# GLOBALS
SENTENCE_LENGTH = 20 # set standard sentence length. Inputs will be truncated or padded
TAGS = ['Fall_risk', 'Msk_prob',  'Nutrition', 'Resp_imp']

batchstring = '03'


outdir = f"{os.getcwd()}/output/"
datadir = f"{os.getcwd()}/data/"
ALdir = f"{outdir}saved_models/AL{batchstring}/"
sheepish_mkdir(f"{ALdir}figures")



########################################
# load the pickles and figure out which configuration is best
cvmods = os.listdir(f"{ALdir}cv_models/")

pkls = [i for i in cvmods if 'model' in i]
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
dfa = dfa.rename(columns = {'brier_all':"mu"})
dfb = dfb.rename(columns = {'brier_all':"sig"})
dfagg = dfa.merge(dfb)
dfagg= dfagg.reset_index(drop = False)
dfagg=dfagg.sort_values('mu')

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
    
meanlen = np.mean([len(i) for i in hdict['L']]) //1
ff = lossplot(hdict, meanlen)
ff.savefig(f"{ALdir}figures/cvlossplot.pdf")
ff.savefig(f"{ALdir}figures/cvlossplot.png", dpi = 400)


##################
# load data
df_tr = pd.read_csv(f"{ALdir}processed_data/trvadata/r1_f0_tr_df.csv", index_col = 0)
df_va = pd.read_csv(f"{ALdir}processed_data/trvadata/r1_f0_va_df.csv", index_col = 0)
df = pd.concat([df_tr, df_va])

train_sent = df['sentence']

str_varnames = [i for i in df.columns if re.match("pca[0-9]",i)]


###################
# create model and vectorizer
cfg = read_pickle(f"{ALdir}cv_models/{bestmods[0]}")['config']
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
    model, vectorizer = make_model(emb_path = f"{datadir}w2v_oa_all_300d.bin",
                                   sentence_length = SENTENCE_LENGTH,
                                   meta_shape = len(str_varnames),
                                   tags = TAGS,
                                   train_sent = train_sent,
                                   l1_l2_pen = cfg['l1_l2_pen'],
                                   n_units = cfg['n_units'],
                                   n_dense = cfg['n_dense'],
                                   dropout = cfg['dropout'])
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(1e-4))


#####################
# prepare data for tensorfow
tr_text = vectorizer(np.array([[s] for s in train_sent]))

tr_labels = []
for n in TAGS:
    tr = tf.convert_to_tensor(df_tr[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype = 'float32')
    tr_labels.append(tr)

tr_struc = tf.convert_to_tensor(df[str_varnames], dtype = 'float32')

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


plt.plot([1])
