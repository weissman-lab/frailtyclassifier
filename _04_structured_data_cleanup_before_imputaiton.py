


import os
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
import re
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

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
elix['month'] = elix.LATEST_TIME.dt.month + (elix.LATEST_TIME.dt.year - 2018) * 12
elix = elix.drop(columns=['CSNS', 'LATEST_TIME', 'Unnamed: 0'])
strdat = strdat.merge(elix, how='outer')


# n_comorb from conc_notes_df
conc_notes_df = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (conc_notes_df.LATEST_TIME.dt.year - 2018) * 12
mm = conc_notes_df[['PAT_ID', 'month', 'n_comorb']]
strdat = strdat.merge(mm, how='outer')

rawnotes = pd.read_pickle(f"{datadir}raw_notes_df.pkl")
demogs = rawnotes[['PAT_ID', 'AGE', 'SEX', 'MARITAL_STATUS', 'RELIGION',
                   'EMPY_STAT', 'RACE', 'LANGUAGE']].copy() #, 'COUNTY']]
demogs.AGE = demogs.AGE.astype(float)
demogs.loc[demogs.MARITAL_STATUS == 'Domestic Partner', 'MARITAL_STATUS'] = 'Partner'
demogs.loc[demogs.RELIGION == 'Church of Jesus Christ of Latter-day Saints', 'RELIGION'] = 'Mormon'
demogs.loc[~demogs.LANGUAGE.isin(['English', 'Spanish']), 'LANGUAGE'] = 'Other'
demogs['month'] = rawnotes.NOTE_ENTRY_TIME.dt.month + (rawnotes.NOTE_ENTRY_TIME.dt.year - 2018) * 12
# demogs.loc[~demogs.COUNTY.isin(['PHILADELPHIA', 'CHESTER', 'DELAWARE', 'MONTGOMERY', 'BUCKS', 'CAMDEN',
#                                 'GLOUCESTER', 'BURLINGTON', 'MERCER']), 'COUNTY'] = 'other'
demogs = demogs.drop_duplicates(subset=['PAT_ID', 'month'])

strdat = strdat.merge(demogs, how = 'left', on = ['PAT_ID', 'month'])

# re-wrote data cleaning from _04_combine_structured_data.R in python
strdat.loc[strdat.RACE.isin(['Unknown', '']), 'RACE'] = np.nan
strdat.loc[~strdat.RACE.isin(['White', 'Black']) & strdat.RACE.notnull(), 'RACE'] = "Other"
strdat.loc[(strdat.AGE > 100) | (strdat.AGE < 18), 'AGE'] = np.nan
strdat.loc[strdat.SEX == '', 'SEX'] = np.nan
strdat.loc[strdat.EMPY_STAT.isin(['', 'Unknown']), 'EMPY_STAT'] = np.nan
strdat.loc[strdat.EMPY_STAT.isin(['Homemaker', 'Self Employed',
                                 'On Active Military Duty', 'Per Diem',
                                 'Student - Part Time', 'Student - Full Time']),
           'EMPY_STAT'] = "Other"
strdat.loc[strdat.EMPY_STAT == 'Retired Military', 'EMPY_STAT'] = 'Retired'
strdat.loc[strdat.MARITAL_STATUS == '', 'MARITAL_STATUS'] = np.nan
strdat.loc[strdat.MARITAL_STATUS == 'Separated', 'MARITAL_STATUS'] = 'Divorced'
strdat.loc[strdat.MARITAL_STATUS == 'Partner', 'MARITAL_STATUS'] = 'Other'
strdat.loc[strdat.LANGUAGE == '', 'LANGUAGE'] = np.nan
#set dtypes
strdat.SEX = strdat.SEX.astype(str)
strdat.MARITAL_STATUS = strdat.MARITAL_STATUS.astype(str)
strdat.RELIGION = strdat.RELIGION.astype(str)
strdat.EMPY_STAT = strdat.EMPY_STAT.astype(str)
strdat.RACE = strdat.RACE.astype(str)
strdat.LANGUAGE = strdat.LANGUAGE.astype(str)


#convert all missing to np.nan
strdat = strdat.fillna(value=np.nan)

# I don't understand this one
# df$X <- df$RELIGION <- NULL

strdat.to_csv(f"{outdir}structured_data_merged_cleaned.csv")
