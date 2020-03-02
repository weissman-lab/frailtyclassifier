

'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Vitals:  blood pressure and weight

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
vitals
it turns out that BP is in PAT_ENC.  I could have pulled it together with the 
'''
if "vitals_raw.pkl" not in os.listdir(outdir):
    bq = '''
select
        pe.PAT_ID
    ,   pe.PAT_ENC_CSN_ID
    ,   datediff(day, p.BIRTH_DATE, pe.CONTACT_DATE) / 365.0        as AGE
    ,   pe.CONTACT_DATE
    ,   pe.BP_DIASTOLIC
    ,   pe.BP_SYSTOLIC
    ,   pe.WEIGHT
    ,   pe.HEIGHT
from PAT_ENC as pe
join PATIENT as p on p.PAT_ID = pe.PAT_ID
    '''
    fdict = dict(PAT_ID={"vals": [], "foreign_key":"PAT_ID","foreign_table": "pe"})

    def wrapper(ids):
        start = time.time()
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
        q += "where pe.CONTACT_DATE >= '2017-01-01'"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        # remove rows where none of the useful variables are measured
        out = out.loc[(~np.isnan(out.BP_DIASTOLIC)) |
                      (~np.isnan(out.BP_SYSTOLIC)) |
                      (~out.HEIGHT.isnull()) |
                      (~np.isnan(out.WEIGHT))]
        print(f"did file starting with {ids[0]}, and it's got {out.shape}.  it took {time.time()-start}")
        return out


    UIDs = df.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    vitout = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    vitdf = pd.concat(vitout)
    vitdf.to_pickle(f"{outdir}vitals_raw.pkl")
else:
    vitdf = pd.read_pickle(f"{outdir}vitals_raw.pkl")


