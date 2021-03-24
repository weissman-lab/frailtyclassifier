'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Encounters:

'''
import os
import pandas as pd
from _99_project_module import get_clarity_conn, get_from_clarity_then_save,     query_filtered_with_temp_tables, write_txt, read_txt, nrow
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
datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"

# load the concatenated notes DF
df = pd.read_pickle((f'{outdir}conc_notes_df.pkl'))

'''
Encounters
'''
if "encs_raw.pkl" not in os.listdir(datadir):
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
    encdf.to_pickle(f"{datadir}encs_raw.pkl")
else:
    encdf = pd.read_pickle(f"{datadir}encs_raw.pkl")

# Office visit
office_visit = encdf[encdf.ENCOUNTER_TYPE.isin(
    ['Appointment', 'Office Visit', 'Post Hospitalization', 'Post Emergency'])
    & ~encdf.PATIENT_CLASS.isin(
    ['Inpatient', 'Emergency', 'Radiation/Oncology-Recurring',
     'Therapies-Recurring', 'Chemo Series', 'Day Surgery', 'Observation',
     'AM Admit', 'Semi-Private/Med-Surg', 'Endo/Bronch', 'ICU',
     'Hyperbaric-recurring', 'EP/CATH', 'Isolation', 'Partial Hospitalization',
     'CCU (Beds 1-7 AND Beds on 4E S,Q,O)', 'CCU (SCCU Beds 8-13)',
     'Rehab Inpatient', 'Gamma Knife'])]
office_visit['office_visit'] = 1
office_visit['ED_visit'] = 0
office_visit['admission'] = 0

# ED visits
ED_visit = encdf[encdf.ENCOUNTER_TYPE.isin(
    ['Hospital Encounter', 'Emergency Department'])
    & encdf.PATIENT_CLASS.isin(
    ['Emergency', 'Observation'])]
ED_visit['ED_visit'] = 1
ED_visit['office_visit'] = 0
ED_visit['admission'] = 0

# Hospital admissions
admission = encdf[encdf.ENCOUNTER_TYPE.isin(
    ['Hospital Encounter', 'Emergency Department'])
    & encdf.PATIENT_CLASS.isin(
    ['Inpatient', 'ICU', 'Semi-Private/Med-Surg', ""])]
admission['admission'] = 1
admission['office_visit'] = 0
admission['ED_visit'] = 0

# concat
encdf = pd.concat([office_visit, ED_visit, admission], ignore_index=True)

## Aggregation
'''
Now, build a dataset that builds 6m numbers of visits, ED visits, admissions, and total time hospitalized
'''
# pull the most recent CSN from the concatenated notes df
df['PAT_ENC_CSN_ID'] = df.CSNS.apply(lambda x: int(x.split(",")[0]))
df['month'] = df.LATEST_TIME.dt.month+(df.LATEST_TIME.dt.year-2018)*12

# write a function that takes a row of the concatenated notes DF and outputs a dict of lab values
def recent_encs(i):
    try:
        edf = encdf.loc[(encdf.PAT_ID == df.PAT_ID.iloc[i]) &
                        (encdf.CONTACT_DATE <= df.LATEST_TIME.iloc[i] + pd.DateOffset(
                            days=1)) &  # the deals with the fact that the times might be rounded down to the nearest day sometimes
                        (encdf.CONTACT_DATE >= df.LATEST_TIME.iloc[i] - pd.DateOffset(months=6))
                        ]
        if nrow(edf) > 0:
            outdict = dict(PAT_ID = df.PAT_ID.iloc[i],
                           PAT_ENC_CSN_ID=df.PAT_ENC_CSN_ID.iloc[i],
                           month = df.month.iloc[i],
                           n_encs = nrow(edf),
                           n_ed_visits = int(edf.ED_visit.sum()),
                           n_admissions = int(edf.admission.sum()))
            if any(~edf.HOSP_ADMSN_TIME.isna()):
                outdict['days_hospitalized'] = (edf.HOSP_DISCH_TIME - edf.HOSP_ADMSN_TIME).sum().total_seconds()/60/60/24
            else:
                outdict['days_hospitalized'] = 0
            return outdict
        else:
            return
    except Exception as e:
        print(f"======={i}======")
        print(e)
        return e


pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
rencs = pool.map(recent_encs, range(nrow(df)), chunksize=1)
print(time.time() - start)
pool.close()


errs = [i for i in range(len(rencs)) if type(rencs[i]).__name__ != "dict"]
rencs_fixed = [i for i in rencs if type(i).__name__ == "dict"]

encs_6m = pd.DataFrame(rencs_fixed)

encs_6m.to_pickle(f"{outdir}encs_6m.pkl")

