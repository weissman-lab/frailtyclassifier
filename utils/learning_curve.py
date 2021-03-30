from utils.misc import read_pickle
import os
import pandas as pd
from utils.organization import find_outdir
from utils.constants import TAGS

outdir = find_outdir()

batches = ['AL01', 'AL02', 'AL03', 'AL04']
mult_b = []
single_b = []
def learning_curve():
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

if __name__ == "__main__":
    main()