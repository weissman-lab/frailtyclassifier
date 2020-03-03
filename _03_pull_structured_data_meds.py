
'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Meds:

'''
import os
os.chdir('/Users/crandrew/projects/GW_PAIR_frailty_classifier/')
import pandas as pd
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, \
    query_filtered_with_temp_tables, write_txt, read_txt, nrow
import re
import time
import multiprocessing as mp
import copy
import numpy as np


# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# connect to the database
clar_conn = get_clarity_conn("/Users/crandrew/Documents/clarity_creds_ACD.yaml")

# set a global data dir for PHI
datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"

# load the concatenated notes DF
df = pd.read_pickle((f'{outdir}conc_notes_df.pkl'))


'''
meds
'''
if "meds_raw.pkl" not in os.listdir(outdir):
    bq = '''
select
        om.PAT_ID
    ,   om.PAT_ENC_CSN_ID
    ,   om.ORDERING_DATE
    ,   om.MEDICATION_ID
    ,   cm.NAME
from ORDER_MED as om
join CLARITY_MEDICATION as cm on om.MEDICATION_ID = cm.MEDICATION_ID
    '''
    fdict = dict(PAT_ID={"vals": [], "foreign_key":"PAT_ID","foreign_table": "om"})

    def wrapper(ids):
        start = time.time()
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
        q += "where om.ORDERING_DATE >= '2017-01-01'"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        print(f"did file starting with {ids[0]}, and it's got {out.shape}.  it took {time.time()-start}")
        return out


    UIDs = df.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    medout = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    meddf = pd.concat(medout)
    meddf.to_pickle(f"{outdir}meds_raw.pkl")
else:
    meddf = pd.read_pickle(f"{outdir}meds_raw.pkl")


# aggregate by taking number of unique rx per period

# pull the most recent CSN from the concatenated notes df
df['PAT_ENC_CSN_ID'] = df.CSNS.apply(lambda x: int(x.split(",")[0]))
df['month'] = df.LATEST_TIME.dt.month+(df.LATEST_TIME.dt.year-2018)*12

# write a function that takes a row of the concatenated notes DF and outputs a dict of lab values
def recent_meds(i):
    try:
        mdf = meddf.loc[(meddf.PAT_ID == df.PAT_ID.iloc[i]) &
                        (meddf.ORDERING_DATE <= df.LATEST_TIME.iloc[i] + pd.DateOffset(
                            days=1)) &  # the deals with the fact that the times might be rounded down to the nearest day sometimes
                        (meddf.ORDERING_DATE >= df.LATEST_TIME.iloc[i] - pd.DateOffset(months=6))
                        ]
        if nrow(mdf) > 0:
            outdict = dict(PAT_ID = df.PAT_ID.iloc[i],
                           PAT_ENC_CSN_ID=df.PAT_ENC_CSN_ID.iloc[i],
                           month=df.month.iloc[i],
                           n_unique_meds = mdf.MEDICATION_ID.nunique())
            return outdict
        else:
            return
    except Exception as e:
        print(f"======={i}======")
        print(e)
        return e



pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
rmeds = pool.map(recent_meds, range(nrow(df)), chunksize=1)
print(time.time() - start)
pool.close()


errs = [i for i in range(len(rmeds)) if type(rmeds[i]).__name__ != "dict"]
rmeds_fixed = [i for i in rmeds if type(i).__name__ == "dict"]

meds_6m = pd.DataFrame(rmeds_fixed)

meds_6m.to_pickle(f"{outdir}meds_6m.pkl")
