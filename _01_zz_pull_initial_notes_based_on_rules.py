
'''
This is a scratch-ish script to pull some initial notes based on rules.
It'll look in the `znotes` and the `notes text` data frames to pull:
- 50 high-prob notes
- 50 low-prob notes
- 50 random notes
Feedback from review of those notes could lead us to go back and change the queries,
before building other stuff later.
'''

import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
import os
import json
import gc

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/"
# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load "znotes" -- the notes metadata
znotes = pd.read_json(f"{datadir}znotes.json.bz2")
znotes = znotes[znotes['ENTRY_TIME'].dt.year >= 2018]

# next load all of the diagnoses, so that we can merge them onto znotes
dx_df = pd.read_json(f'{datadir}all_dx_df.json.bz2')
dx_df.drop_duplicates(inplace=True)

# make sure no commas in the dx df -- this would imply multiple dx per line
has_commas = dx_df[dx_df['CODE'].str.contains(",")]
print(str(has_commas.shape[0]) + " diagnoses have commas")

# we need to associate each note with the diagnoses from the past 12 month.
# I'll do this by first grouping the diagnosis df by patient ID
# then, for each note, get the relevant date, and look up all of the diagnoses within the relevant range of that date
# this should be doable in parallel
dx_groups = dx_df.groupby(by = dx_df.PAT_ID)
UID_dx = list(set(list(dx_groups.groups.keys())))

# write a function that takes a data frame and a date, and returns last-12-month morbidities
# it takes a dict with elements "date" and "df".
# date is the end of the 12-month window, and "df" are all the diagnoses for that particular patient
def get_dx_within_window(i, window = 1):
    inp = {"df": dx_groups.get_group(znotes.PAT_ID[znotes.PAT_ENC_CSN_ID == i].iloc[0]).drop_duplicates(),
           "date": znotes.ENTRY_TIME[znotes.PAT_ENC_CSN_ID == i]}
    timediff = (inp["date"].iloc[0] - inp["df"].ENTRY_TIME).dt.total_seconds() / 60 / 60 / 25 / 365
    dx_one_year = inp["df"].loc[(timediff > 0) & (timediff < window)].CODE.unique()
    outdict = {"PAT_ENC_CSN_ID" : int(inp["df"].PAT_ENC_CSN_ID.iloc[0]),
               "DX" : ','.join(dx_one_year)}
    with open(f'{datadir}/output/_i{i}data.json', 'w') as fp:
        json.dump(outdict, fp)
    del inp
    del timediff
    del dx_one_year
    del outdict
    gc.collect()

# make a list of CSNs to feed to the get_dx_within_window function
lgen = znotes.PAT_ENC_CSN_ID[znotes.PAT_ID.isin(UID_dx)].tolist()

pool = multiprocessing.Pool()
pool.map(get_dx_within_window, lgen)

