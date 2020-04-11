


import os
import pandas as pd

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
if 'crandrew' in os.getcwd():
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
from _99_project_module import inv_logit, send_message_to_slack
import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate, Conv1D, \
    LeakyReLU, BatchNormalization
from tensorflow.keras import Model, Input, backend
import pickle
import time

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"
figdir = f"{os.getcwd()}/figures/"


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
                   'EMPY_STAT', 'RACE', 'LANGUAGE', 'COUNTY']]
demogs.AGE = demogs.AGE.astype(float)
demogs.loc[demogs.MARITAL_STATUS == 'Domestic Partner', 'MARITAL_STATUS'] = 'Partner'
demogs.loc[demogs.RELIGION == 'Church of Jesus Christ of Latter-day Saints', 'RELIGION'] = 'Mormon'
demogs.loc[~demogs.LANGUAGE.isin(['English', 'Spanish']), 'LANGUAGE'] = 'Other'
demogs['month'] = rawnotes.NOTE_ENTRY_TIME.dt.month + (rawnotes.NOTE_ENTRY_TIME.dt.year - 2018) * 12
demogs.loc[~demogs.COUNTY.isin(['PHILADELPHIA', 'CHESTER', 'DELAWARE', 'MONTGOMERY', 'BUCKS', 'CAMDEN',
                                'GLOUCESTER', 'BURLINGTON', 'MERCER']), 'COUNTY'] = 'other'
demogs = demogs.drop_duplicates(subset=['PAT_ID', 'month'])

strdat = strdat.merge(demogs, how = 'left', on = ['PAT_ID', 'month'])
strdat.to_csv(f"{outdir}structured_data_merged_partially_cleaned.csv")