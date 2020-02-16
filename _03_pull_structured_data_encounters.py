
'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Encounters:

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
Encounters
'''
if "encs_raw.pkl" not in os.listdir(outdir):
    bq = '''
select
        pe2.PAT_ID
    ,   pe2.PAT_ENC_CSN_ID
    ,   pe2.CONTACT_DATE
    ,   zpc.NAME as PATIENT_CLASS
    ,   zdet.NAME as ENCOUNTER_TYPE
    ,   peh.ADT_ARRIVAL_TIME
    ,   peh.HOSP_ADMSN_TIME
    ,   peh.HOSP_DISCH_TIME
from PAT_ENC_2 as pe2
join PAT_ENC as pe on pe.PAT_ENC_CSN_ID = pe2.PAT_ENC_CSN_ID
left join ZC_PAT_CLASS as zpc on pe2.ADT_PAT_CLASS_C = zpc.ADT_PAT_CLASS_C
left join PAT_ENC_HSP as peh on peh.PAT_ENC_CSN_ID = pe2.PAT_ENC_CSN_ID
left join ZC_DISP_ENC_TYPE as zdet on pe.ENC_TYPE_C = zdet.DISP_ENC_TYPE_C
    '''
    fdict = dict(PAT_ID={"vals": [], "foreign_key":"PAT_ID","foreign_table": "pe"})

    def wrapper(ids):
        start = time.time()
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
        q += "where pe2.CONTACT_DATE >= '2017-01-01'"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        print(f"did file starting with {ids[0]}, and it's got {out.shape}.  it took {time.time()-start}")
        return out


    UIDs = df.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    encout = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    encdf = pd.concat(encout)
    encdf.to_pickle(f"{outdir}encs_raw.pkl")
else:
    encdf = pd.read_pickle(f"{outdir}encs_raw.pkl")


