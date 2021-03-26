from utils.misc import read_pickle
import os
import pandas as pd
from utils.organization import find_outdir

outdir = find_outdir()

batches = ['AL03', 'AL04']
res = []

for batch in batches:
    pklpath = f'{outdir}saved_models/{batch}/cv_models/'
    pkls = [i for i in os.listdir(pklpath) if 'model' in i]
    xd = []
    for pkl in pkls:
        x = read_pickle(f"{pklpath}{pkl}")
        d = x['config']
        d = {**d, **x['brier_classwise']}
        d = {**d, **x['brier_aspectwise']}
        d['brier_all'] = x['brier_all']
        d['runtime'] = x['runtime']
        d['ran_when'] = x['ran_when']
        xd.append(d)
    df = pd.DataFrame(xd)
    df.shape
    # mean of multi-class scaled briers for all aspects
    df['brier_mean_aspects'] = df.loc[:,['Fall_risk', 'Msk_prob', 'Nutrition', 'Resp_imp']].mean(axis = 1)
    # summarize by repeat/fold
    dfagg = df.groupby(['n_dense', 'n_units', 'dropout', 'l1_l2_pen', 'use_case_weights'], as_index=False).agg(
        aspect_mean_all=('brier_mean_aspects', 'mean'),
        aspect_se_all=('brier_mean_aspects', 'sem'),
        count = ('brier_mean_aspects', 'count')
    )
    dfagg['batch'] = batch
    res.append(dfagg)

lc = pd.concat(res).sort_values(by='aspect_mean_all',
                                ascending=False).reset_index()

lc.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve.csv")