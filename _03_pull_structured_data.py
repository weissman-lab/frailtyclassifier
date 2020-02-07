

'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Labs:  via the labs query.  Here's the list:
Creatinine, BUN (urea nitrogen), sodium, potassium, calcium, magnesium, phosphate (phosphorus), TSH, Aspartate aminotransferase (AST), alkaline phosphatase, total protein (protein, total), albumin, total bilirubin (bilirubin, total), Hemoglobin concentration (hemoglobin), hematocrit, MCV, MCHC, red blood cell distribution width (RDW), platelet concentration (platelets), white blood cells, Prothrombin time (PT), partial thromboplastin time (PTT), Mean systolic BP, mean diastolic BP, Hemoglobin A1c, carbon dioxide, iron, ferritin, transferrin, transferrin sat, LDL cholesterol

'''

import pandas as pd
import os
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, combine_notes_by_type, \
    query_filtered_with_temp_tables, write_txt
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
Labs: there are many, many names for similar labs.
Pull all labs for the patients in our sample, and use this to map all of the synonymous lab names to common names
'''
fdict = dict(PAT_ID={"vals": [], "foreign_table":"pe",
                          "foreign_key":"PAT_ID"})
bq = '''
select distinct
cc.COMMON_NAME
from CLARITY_COMPONENT as cc
join ORDER_RESULTS as ores on ores.COMPONENT_ID = cc.COMPONENT_ID
join PAT_ENC as pe on pe.PAT_ENC_CSN_ID = ores.PAT_ENC_CSN_ID
'''

def wrapper(ids):
    fd = copy.deepcopy(fdict)
    fd['PAT_ID']['vals'] = ids
    q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
    q += "where pe.ENTRY_TIME >= '2017-01-01'"
    out = get_from_clarity_then_save(q, clar_conn=clar_conn)
    return out

UIDs = df.PAT_ID.unique().tolist()
UIDs.sort()
chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]


pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
labnames = pool.map(wrapper, chunks, chunksize=1)
print(time.time() - start)
pool.close()

labs = pd.concat(labnames)
labs = labs.sort_values("COMMON_NAME")
pd.Series(labs.COMMON_NAME.unique().tolist()).to_csv(f'{outdir}/labnames.csv')


