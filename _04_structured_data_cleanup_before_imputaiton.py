import os

import numpy as np
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"

start = True
# output data for R
for i in [i for i in os.listdir(outdir) if "_6m" in i]:
    x = pd.read_pickle(f"{outdir}{i}")
    x = x.drop(columns="PAT_ENC_CSN_ID")
    if start is True:
        start = False
        strdat = x
    else:
        strdat = strdat.merge(x, how='outer')
    print(strdat.shape)

# elixhauser
elix = pd.read_csv(f"{outdir}elixhauser_scores.csv")
elix.LATEST_TIME = pd.to_datetime(elix.LATEST_TIME)
elix['month'] = elix.LATEST_TIME.dt.month + (
            elix.LATEST_TIME.dt.year - 2018) * 12
elix = elix.drop(columns=['CSNS', 'LATEST_TIME', 'Unnamed: 0'])
strdat = strdat.merge(elix, how='outer')

# n_comorb from conc_notes_df
conc_notes_df = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (
            conc_notes_df.LATEST_TIME.dt.year - 2018) * 12
mm = conc_notes_df[['PAT_ID', 'month', 'n_comorb']]
strdat = strdat.merge(mm, how='outer')

# Including race & language here, but will remove from model features
rawnotes = pd.read_pickle(f"{datadir}raw_notes_df.pkl")
demogs = rawnotes[['PAT_ID', 'AGE', 'SEX', 'MARITAL_STATUS', 'EMPY_STAT',
                   'RACE', 'LANGUAGE']].copy()  # 'RELIGION', 'COUNTY']]
demogs.AGE = demogs.AGE.astype(float)
demogs.loc[
    demogs.MARITAL_STATUS == 'Domestic Partner', 'MARITAL_STATUS'] = 'Partner'
demogs.loc[~demogs.LANGUAGE.isin(['English', 'Spanish']), 'LANGUAGE'] = 'Other'
demogs['month'] = rawnotes.NOTE_ENTRY_TIME.dt.month + (
            rawnotes.NOTE_ENTRY_TIME.dt.year - 2018) * 12
demogs = demogs.drop_duplicates(subset=['PAT_ID', 'month'])

strdat = strdat.merge(demogs, how='left', on=['PAT_ID', 'month'])

# re-wrote data cleaning from _04_combine_structured_data.R in python
strdat.loc[strdat.RACE.isin(['Unknown', '']), 'RACE'] = np.nan
strdat.loc[~strdat.RACE.isin(
    ['White', 'Black']) & strdat.RACE.notnull(), 'RACE'] = "Other"
strdat.loc[(strdat.AGE > 100) | (strdat.AGE < 18), 'AGE'] = np.nan
strdat.loc[strdat.SEX == '', 'SEX'] = np.nan
strdat.loc[strdat.EMPY_STAT.isin(['', 'Unknown']), 'EMPY_STAT'] = np.nan
strdat.loc[strdat.EMPY_STAT.isin(['Homemaker', 'Self Employed',
                                  'On Active Military Duty', 'Per Diem',
                                  'Student - Part Time',
                                  'Student - Full Time']),
           'EMPY_STAT'] = "Other"
strdat.loc[strdat.EMPY_STAT == 'Retired Military', 'EMPY_STAT'] = 'Retired'
strdat.loc[strdat.MARITAL_STATUS == '', 'MARITAL_STATUS'] = np.nan
strdat.loc[strdat.MARITAL_STATUS == 'Separated', 'MARITAL_STATUS'] = 'Divorced'
strdat.loc[strdat.MARITAL_STATUS == 'Partner', 'MARITAL_STATUS'] = 'Other'
strdat.loc[strdat.LANGUAGE == '', 'LANGUAGE'] = np.nan

# convert all missing to np.nan
strdat = strdat.fillna(value=np.nan)
# make data frame of missing values
strdat_MV = strdat.loc[:, ~strdat.columns.isin(['PAT_ID', 'month'])].copy()
for c in strdat_MV.columns:
    strdat_MV[c] = strdat_MV[c].isnull()
    if strdat_MV[c].sum() == len(strdat_MV[c]):
        strdat_MV[c] = np.nan
#rename cols
strdat_MV.columns = ['MV_' + n for n in strdat_MV.columns]

#make dummies for categoricals
#don't need dums for race & language because they wont be in the model
dumcols = ['SEX', 'MARITAL_STATUS', 'EMPY_STAT']
dums = pd.concat(
    [pd.get_dummies(strdat[[i]]) for i in dumcols], axis=1)

#combine and write out
strdat_all = pd.concat([strdat.loc[:, ~strdat.columns.isin(dumcols)],
                        dums,
                        strdat_MV], axis=1)
strdat_all.to_csv(f"{outdir}structured_data_merged_cleaned.csv")
