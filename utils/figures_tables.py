from utils.misc import read_pickle
import os
import pandas as pd
from utils.organization import find_outdir
import numpy as np
from utils.constants import TAGS
import re


def table_1_demographics():
    '''
    Makes table 1 with comparison of training and test notes
     '''
    # set directories
    outdir = find_outdir()
    #load train and test notes
    notes_train = pd.read_csv(
        f"{outdir}saved_models/AL05/processed_data/full_set/full_df.csv")
    notes_test = pd.read_csv(
        f"{outdir}saved_models/AL05/processed_data/test_set/full_df.csv")
    #get month
    notes_train['month'] = [int(m.split('_')[-3].split('m')[1]) for m in notes_train.sentence_id]
    notes_test['month'] = [int(m.split('_')[-3].split('m')[1]) for m in notes_test.sentence_id]
    notes_train = notes_train[['PAT_ID', 'month']].drop_duplicates()
    notes_test = notes_test[['PAT_ID', 'month']].drop_duplicates()
    # load structured data & match to train and test data
    strdat = pd.read_csv(f"{outdir}structured_data_merged_cleaned.csv",
                         index_col=0)
    strdat_train = strdat.merge(notes_train[['PAT_ID', 'month']])
    strdat_test = strdat.merge(notes_test[['PAT_ID', 'month']])
    strdat_train['train_test'] = 'train'
    strdat_test['train_test'] = 'test'
    all_str = pd.concat([strdat_train, strdat_test])
    # fix encounters in structured_data_merged_cleaned.csv with encs_6m.pkl.
    fix_encs = pd.read_pickle(f"{outdir}encs_6m.pkl")
    fix_encs['f_n_encs'] = fix_encs['n_encs']
    fix_encs['f_n_ed_visits'] = fix_encs['n_ed_visits']
    fix_encs['f_n_admissions'] = fix_encs['n_admissions']
    fix_encs['f_days_hospitalized'] = fix_encs['days_hospitalized']
    fix_encs = fix_encs.drop(['n_encs', 'n_ed_visits', 'n_admissions', 'days_hospitalized'], axis =1)
    all_str = all_str.merge(fix_encs, 'left', on = ['PAT_ID', 'month'])
    # get number of notes per patient from cndf
    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    all_str = all_str.merge(cndf[['PAT_ID', 'month', 'n_notes']], 'left', on = ['PAT_ID', 'month'])
    #mean, sd
    mean_cols = []
    #median, IQR
    median_cols = ['n_notes', 'mean_sys_bp', 'mean_dia_bp', 'bmi_mean', 'bmi_slope',
                 'spo2_worst', 'ALBUMIN',  'CALCIUM', 'CO2', 'HEMATOCRIT',
                 'HEMOGLOBIN', 'LDL', 'MCHC', 'MCV',  'POTASSIUM', 'PROTEIN',
                 'RDW', 'SODIUM', 'WBC', 'MAGNESIUM', 'TRANSFERRIN',
                 'TRANSFERRIN_SAT', 'PHOSPHATE', 'elixhauser',
                   'AGE', 'f_n_encs', 'max_o2', 'ALKALINE_PHOSPHATASE',
                   'AST', 'BILIRUBIN', 'BUN', 'CREATININE', 'PLATELETS',
                   'n_ALBUMIN', 'n_ALKALINE_PHOSPHATASE', 'n_AST', 'n_BILIRUBIN',
                   'n_BUN', 'n_CALCIUM', 'n_CO2', 'n_CREATININE', 'n_HEMATOCRIT',
                   'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
                   'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC',
                   'n_FERRITIN', 'n_IRON', 'n_MAGNESIUM', 'n_TRANSFERRIN',
                   'n_TRANSFERRIN_SAT', 'n_PT', 'n_PHOSPHATE', 'n_PTT', 'FERRITIN',
                   'IRON', 'PT', 'PTT', 'n_unique_meds', 'n_comorb']
    #median, IQR, range
    range_cols = ['f_n_ed_visits', 'f_n_admissions',
                   'f_days_hospitalized']
    #n, %
    n_cols = ['Race_white', 'Race_black', 'Race_other', 'Language_english',
              'Language_spanish', 'SEX_Female', 'SEX_Male',
              'MARITAL_STATUS_Divorced', 'MARITAL_STATUS_Married',
              'MARITAL_STATUS_Other', 'MARITAL_STATUS_Single',
              'MARITAL_STATUS_Widowed', 'EMPY_STAT_Disabled',
              'EMPY_STAT_Full Time', 'EMPY_STAT_Not Employed', 'EMPY_STAT_Other',
              'EMPY_STAT_Part Time', 'EMPY_STAT_Retired']
    #new dums
    all_str['Race_white'] = np.where(all_str.RACE == 'White', 1, 0)
    all_str['Race_black'] = np.where(all_str.RACE == 'Black', 1, 0)
    all_str['Race_other'] = np.where(all_str.RACE == 'Other', 1, 0)
    all_str['Language_english'] = np.where(all_str.LANGUAGE == 'English', 1, 0)
    all_str['Language_spanish'] = np.where(all_str.LANGUAGE == 'Spanish', 1, 0)
    # drop extra columns
    drop_cols = ['sd_', 'MV_']
    drop_cols = [i for i in all_str.columns if any(xi in i for xi in drop_cols)]
    all_str = all_str.loc[:, ~all_str.columns.isin(drop_cols)]
    all_str.to_csv(f"{outdir}figures_tables/patient_struc_data.csv")
    drop_cols = ['PAT_ID', 'month']
    drop_cols = [i for i in all_str.columns if any(xi in i for xi in drop_cols)]
    all_str = all_str.loc[:, ~all_str.columns.isin(drop_cols)]
    #summarize median (IQR)
    train_median = all_str[all_str.train_test == 'train'][median_cols].median().reset_index()
    train_median.columns = ['val', 'median']
    train_median['train'] = train_median['median'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'train'][median_cols].quantile(0.25).reset_index()[0.25].round(2).astype(str) +\
                              '-' + all_str[all_str.train_test == 'train'][median_cols].quantile(0.75).reset_index()[0.75].round(2).astype(str)+\
                              ')'
    train_median['field'] = train_median['val'] + ', median (IQR)'
    test_median = all_str[all_str.train_test == 'test'][median_cols].median().reset_index()
    test_median.columns = ['val', 'median']
    test_median['test'] = test_median['median'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'test'][median_cols].quantile(0.25).reset_index()[0.25].round(2).astype(str) +\
                              '-' + all_str[all_str.train_test == 'test'][median_cols].quantile(0.75).reset_index()[0.75].round(2).astype(str)+\
                              ')'
    test_median['field'] = test_median['val'] + ', median (IQR)'
    all_median = train_median[['field', 'train']].merge(test_median[['field', 'test']])
    #summarize median (IQR, range)
    train_range = all_str[all_str.train_test == 'train'][range_cols].median().reset_index()
    train_range.columns = ['val', 'median']
    train_range['train'] = train_range['median'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'train'][range_cols].quantile(0.25).reset_index()[0.25].round(2).astype(str) +\
                              '-' + all_str[all_str.train_test == 'train'][range_cols].quantile(0.75).reset_index()[0.75].round(2).astype(str) +\
                              ', ' + all_str[all_str.train_test == 'train'][range_cols].min().reset_index()[0].astype(str) +\
                              '-' + all_str[all_str.train_test == 'train'][range_cols].max().reset_index()[0].round(1).astype(str) +\
                              ')'
    train_range['field'] = train_range['val'] + ', median (IQR, range)'

    test_range = all_str[all_str.train_test == 'test'][range_cols].median().reset_index()
    test_range.columns = ['val', 'median']
    test_range['test'] = test_range['median'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'test'][range_cols].quantile(0.25).reset_index()[0.25].round(2).astype(str) +\
                              '-' + all_str[all_str.train_test == 'test'][range_cols].quantile(0.75).reset_index()[0.75].round(2).astype(str)+\
                              ', ' + all_str[all_str.train_test == 'test'][range_cols].min().reset_index()[0].astype(str) +\
                              '-' + all_str[all_str.train_test == 'test'][range_cols].max().reset_index()[0].round(1).astype(str) +\
                              ')'
    test_range['field'] = test_range['val'] + ', median (IQR, range)'
    all_range = train_range[['field', 'train']].merge(test_range[['field', 'test']])
    #summarize n
    train_n = all_str[all_str.train_test == 'train'][n_cols].sum().reset_index()
    train_n.columns = ['val', 'n']
    train_n_patients = len(all_str[all_str.train_test == 'train'])
    train_n['train'] = train_n['n'].astype(str) + ' (' +\
                       (all_str[all_str.train_test == 'train'][n_cols].sum().reset_index()[0] / train_n_patients * 100).round().astype(int).astype(str) +\
        '%)'
    train_n['field'] = train_n['val'] + ', n (%)'
    test_n = all_str[all_str.train_test == 'test'][n_cols].sum().reset_index()
    test_n.columns = ['val', 'n']
    test_n_patients = len(all_str[all_str.train_test == 'test'])
    test_n['test'] = test_n['n'].astype(str) + ' (' +\
                       (all_str[all_str.train_test == 'test'][n_cols].sum().reset_index()[0] / test_n_patients * 100).round().astype(int).astype(str) +\
        '%)'
    test_n['field'] = test_n['val'] + ', n (%)'
    all_n = train_n[['field', 'train']].merge(test_n[['field', 'test']])
    all_str2 = pd.concat([all_median, all_n, all_range]).reset_index()
    all_str2.to_csv(f"{outdir}figures_tables/table_1.csv")

    #missingness in train data
    train_miss = all_str[all_str.train_test == 'train']
    percent_missing = train_miss.isnull().sum() * 100 / len(train_miss)
    train_miss = pd.DataFrame({'column_name': train_miss.columns,
                                     '% missing in training set': percent_missing})
    #missingness in test data
    test_miss = all_str[all_str.train_test == 'test']
    percent_missing = test_miss.isnull().sum() * 100 / len(test_miss)
    test_miss = pd.DataFrame({'column_name': test_miss.columns,
                                     '% missing in test set': percent_missing})
    missingness = train_miss.merge(test_miss)
    missingness.to_csv(f"{outdir}figures_tables/missingness.csv")

    return(all_str, all_str2, missingness)


'''
consolidate single- and multi-task NN training performance and compute time
run separately for each embedding strategy (w2v, bert, roberta). The compute 
time is needed in order to make a table in utils/figures_tables.R
'''
def consolidate_NN_perf(model):
    batches = ['AL01', 'AL02', 'AL03', 'AL04', 'AL05']
    mult_b = []
    single_b = []
    outdir = find_outdir()
    for batch in batches:
        if model == 'w2v':
            pklpath = f'{outdir}saved_models/{batch}/cv_models/'
        elif model == 'roberta':
            pklpath = f'{outdir}saved_models/{batch}/cv_models/roberta/'
        elif model == 'bert':
            pklpath = f'{outdir}saved_models/{batch}/cv_models/bioclinicalbert/'
        pkls = [i for i in os.listdir(pklpath) if 'model' in i]
        xd = []
        #ID single- and multi-task NN
        multi = [pkl for pkl in pkls if not any(s in pkl for s in TAGS)]
        single = [pkl for pkl in pkls if any(s in pkl for s in TAGS)]
        #gather multi-task NN cross validation performance
        for pkl in multi:
            x = read_pickle(f"{pklpath}{pkl}")
            d = x['config']
            d = {**d, **x['brier_classwise']}
            d = {**d, **x['brier_aspectwise']}
            d['brier_all'] = x['brier_all']
            d['runtime'] = x['runtime']
            d['ran_when'] = x['ran_when']
            d['pkl_dig'] = re.findall('\d+', pkl)[0]
            xd.append(d)
        multi = pd.DataFrame(xd)
        multi.shape
        # mean of multi-class scaled briers for all aspects
        multi['brier_mean_aspects'] = multi.loc[:, ['Fall_risk', 'Msk_prob', 'Nutrition', 'Resp_imp']].mean(axis = 1)
        multi['batch'] = batch
        mult_b.append(multi)
        # gather single-task NN cross validation performance
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
    mtask = pd.concat(mult_b).reset_index()
    if model == 'w2v':
        mtask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_mtask.csv")
    elif model == 'roberta':
        mtask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_mtask_roberta.csv")
    elif model == 'bert':
        mtask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_mtask_bert.csv")
    #write out
    stask = pd.concat(single_b).reset_index()
    if model == 'w2v':
        stask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_stask.csv")
    elif model == 'roberta':
        stask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_stask_roberta.csv")
    elif model == 'bert':
        stask.to_csv(f"{outdir}saved_models/{batches[-1]}/learning_curve_stask_bert.csv")


def lossplot(d, meanlen, d_final = None):
    import matplotlib.pyplot as plt
    from utils.misc import arraymaker
    import numpy as np
    fig = plt.figure(constrained_layout=True, figsize=(10,4))
    gs = fig.add_gridspec(2, 4)
    # overall
    f_ax1 = fig.add_subplot(gs[:2, :2])
    L = arraymaker(d, 'L')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax1.plot(xg, mu, label = 'CV mean')
    if d_final is not None:
        f_ax1.plot(xg[:len(d_final['L'])], d_final['L'], label = 'final training')
    f_ax1.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax1.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax1.legend()
    f_ax1.set_title('Overall loss')
    # fall Risk
    f_ax2 = fig.add_subplot(gs[0, 2])
    L = arraymaker(d, 'fL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax2.plot(xg, mu)
    if d_final is not None:
        f_ax2.plot(xg[:len(d_final['fL'])], d_final['fL'])
    f_ax2.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax2.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax2.set_title('Fall risk loss')
    # msk
    f_ax3 = fig.add_subplot(gs[0, 3])
    L = arraymaker(d, 'mL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax3.plot(xg, mu)
    if d_final is not None:
        f_ax3.plot(xg[:len(d_final['mL'])], d_final['mL'])
    f_ax3.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax3.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax3.set_title('MSK prob loss')
    # nut
    f_ax4 = fig.add_subplot(gs[1, 2])
    L = arraymaker(d, 'nL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax4.plot(xg, mu)
    if d_final is not None:
        f_ax4.plot(xg[:len(d_final['nL'])], d_final['nL'])
    f_ax4.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax4.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax4.set_title('Resp Imp loss')
    # resp_imp
    f_ax5 = fig.add_subplot(gs[1, 3])
    L = arraymaker(d, 'rL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax5.plot(xg, mu)
    if d_final is not None:
        f_ax5.plot(xg[:len(d_final['rL'])], d_final['rL'])
    f_ax5.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax5.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax5.set_title('Nutrition loss')
    return fig


if __name__ == "__main__":
    pass