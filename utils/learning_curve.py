from utils.misc import read_pickle
import os
import pandas as pd
from utils.organization import find_outdir
from utils.constants import TAGS

def learning_curve():
    outdir = find_outdir()

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
        for pkl in pkls:
            x = read_pickle(f"{pklpath}{pkl}")
            d = x['config']
            d = {**d, **x['brier_classwise']}
            d = {**d, **x['brier_aspectwise']}
            d['brier_all'] = x['brier_all']
            d['runtime'] = x['runtime']
            d['ran_when'] = x['ran_when']
            xd.append(d)
        multi = pd.DataFrame(xd)
        if pkl in single:
            multi.groupby(
                ['n_dense', 'n_units', 'dropout', 'l1_l2_pen', 'use_case_weights',
                 'repeat', 'fold'],
                as_index=False).first()
            multi['model'] = 'NN_single'
        else:
            multi['model'] = 'NN_multi'
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

    #write out
    lc_nn = pd.concat(mult_b).sort_values(by='brier_mean_aspects_mean',
                                    ascending=False).reset_index()
    lc_nn.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_nn.csv")

if __name__ == "__main__":
    main()