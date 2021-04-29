import pandas as pd
from utils.organization import find_outdir
import numpy as np


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
    # drop extra columns
    drop_cols = ['PAT_ID', 'month', 'sd_', 'MV_']
    drop_cols = [i for i in all_str.columns if any(xi in i for xi in drop_cols)]
    all_str = all_str.loc[:, ~all_str.columns.isin(drop_cols)]
    #mean, sd
    mean_cols = ['AGE', 'mean_sys_bp', 'mean_dia_bp', 'bmi_mean', 'bmi_slope',
                 'spo2_worst', 'ALBUMIN',  'CALCIUM', 'CO2', 'HEMATOCRIT',
                 'HEMOGLOBIN', 'LDL', 'MCHC', 'MCV',  'POTASSIUM', 'PROTEIN',
                 'RDW', 'SODIUM', 'WBC', 'MAGNESIUM', 'TRANSFERRIN',
                 'TRANSFERRIN_SAT', 'PHOSPHATE', 'elixhauser']
    #median, IQR
    median_cols = ['f_n_encs', 'f_n_ed_visits', 'f_n_admissions',
                   'f_days_hospitalized', 'max_o2', 'ALKALINE_PHOSPHATASE',
                   'AST', 'BILIRUBIN', 'BUN', 'CREATININE', 'PLATELETS',
                   'n_ALBUMIN', 'n_ALKALINE_PHOSPHATASE', 'n_AST', 'n_BILIRUBIN',
                   'n_BUN', 'n_CALCIUM', 'n_CO2', 'n_CREATININE', 'n_HEMATOCRIT',
                   'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
                   'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC',
                   'n_FERRITIN', 'n_IRON', 'n_MAGNESIUM', 'n_TRANSFERRIN',
                   'n_TRANSFERRIN_SAT', 'n_PT', 'n_PHOSPHATE', 'n_PTT', 'FERRITIN',
                   'IRON', 'PT', 'PTT', 'n_unique_meds', 'n_comorb']
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
    #summarize mean (SD)
    train_mean = all_str[all_str.train_test == 'train'][mean_cols].mean().reset_index()
    train_mean.columns = ['val', 'mean']
    train_mean['train'] = train_mean['mean'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'train'][mean_cols].std().reset_index()[0].round(2).astype(str) +\
                              ')'
    train_mean['field'] = train_mean['val'] + ', mean (SD)'
    test_mean = all_str[all_str.train_test == 'test'][mean_cols].mean().reset_index()
    test_mean.columns = ['val', 'mean']
    test_mean['test'] = test_mean['mean'].round(2).astype(str) + ' (' +\
                          all_str[all_str.train_test == 'test'][mean_cols].std().reset_index()[0].round(2).astype(str) +\
                              ')'
    test_mean['field'] = test_mean['val'] + ', mean (SD)'
    all_mean = train_mean[['field', 'train']].merge(test_mean[['field', 'test']])
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
    all_str2 = pd.concat([all_mean, all_median, all_n]).reset_index()
    all_str2.to_csv(f"{outdir}figures_tables/table_1.csv")

    #table 1 - summary stats
    def summ_stat(train_test):
        tab1_mean = train_test[mean_cols].mean()
        tab1_sd = train_test[mean_cols].std()
        tab1_mean = pd.concat([tab1_mean, tab1_sd],
                              axis=1,
                              keys=['mean', 'sd'])
        tab1_median = train_test[median_cols].median()
        tab1_iqr25 = train_test[median_cols].quantile(0.25)
        tab1_iqr75 = train_test[median_cols].quantile(0.75)
        tab1_median = pd.concat([tab1_median, tab1_iqr25, tab1_iqr75],
                                axis=1,
                                keys=['median', 'IQR_25', 'IQR_75'])
        tab1_n = train_test[n_cols].sum()
        tab1_percent = train_test[n_cols].sum() / train_test[n_cols].count()
        tab1_n = pd.concat([tab1_n, tab1_percent],
                           axis=1,
                           keys=['n', 'percent'])
        return(pd.concat([tab1_mean, tab1_median, tab1_n]))

    tab1_train = summ_stat(all_str[all_str.train_test == 'train'])
    tab1_test = summ_stat(all_str[all_str.train_test == 'test'])

    tab1_train[tab1_train.index.isin(['AGE', 'Race_white', 'SEX_Female',
                                      'f_n_encs', 'f_days_hospitalized',
                                      'n_unique_meds', 'n_comorb', 'elixhauser'])].\
        to_csv(f"{outdir}figures_tables/table_1_train.csv")
    tab1_test[tab1_test.index.isin(['AGE', 'Race_white', 'SEX_Female',
                                      'f_n_encs', 'f_days_hospitalized',
                                      'n_unique_meds', 'n_comorb', 'elixhauser'])].\
        to_csv(f"{outdir}figures_tables/table_1_test.csv")

    return(tab1_train, tab1_test, all_str, new_AL_str)


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