
'''
This script pulls the data.
It begins by getting a list of all patient IDs who have ever had one of our target diagnoses.
Armed with that list, the next query pulls all of the notes from their qualifying encounters.
Finally, we pull info on transplants, and use that to filter-out encounters that happened post-transplant.
'''

# libraries and imports
import pandas as pd
import os
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, combine_notes_by_type, \
    query_filtered_with_temp_tables
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


# get the diagnoses data frame.  look first in the data directory.
if "diagnosis_df.pkl" in os.listdir(datadir):
    diagnosis_df = pd.read_pickle(f"{datadir}diagnosis_df.pkl")
else:
    diagnosis_df = get_from_clarity_then_save(
        query=open("_8_diagnosis_query.sql").read(),
        clar_conn=clar_conn
    )
    # subset them to the correct set of diagnostic codes
    chronic_regex = '^(J44\.[0,1,9]|J43\.[0,1,2,8,9]|J41\.[0,1,8]|J42|J84\.10|D86|J84|M34\.81|J99\.[0,1])'
    diagnosis_df = diagnosis_df[diagnosis_df['icd10'].str.contains(chronic_regex)]
    diagnosis_df.to_pickle(f"{datadir}diagnosis_df.pkl")

# now get the notes
unique_PIDs = diagnosis_df.PAT_ID.unique().tolist()
unique_PIDs.sort()
base_query = open("_8_patient_notes_query.sql").read()
fdict = dict(PAT_ID={"vals": [], "foreign_table":"pe",
                          "foreign_key":"PAT_ID"},
             SPECIALTY={"vals": ['INTERNAL MEDICINE', 'PULMONARY', 'FAMILY PRACTICE', 'GERONTOLOGY',
                                 'CARDIOLOGY', 'RHEUMATOLOGY', 'Neurology'],
                        "foreign_table": "cd",
                          "foreign_key":"SPECIALTY"},
             NOTE_TYPE={"vals": ['Progress Notes'],
                        "foreign_table": "znti",
                          "foreign_key":"NAME"},
             ENCOUNTER_TYPE={"vals":['Appointment', 'Office Visit', 'Post Hospitalization'],
                             "foreign_table":"zdet",
                          "foreign_key":"NAME"},
             NOTE_STATUS={"vals": ['2', '3'],
                        "foreign_table": "nei",
                          "foreign_key":"note_status_c"}
             )

# download the notes in small batches, using multiprocessing
# save them to a temp directory, and then concatenate them when done
# to enable this to be restarted, let the function first check and see what's already been done
def get_tbd():
    already_got = [i for i in os.listdir(f"{outdir}/tmp/") if 'notes_' in i]
    gotlist = []
    for i in already_got:
        got = re.sub("notes_|\.pkl", "", i).split("_")
        for j in got:
            gotlist += [j]
    tbd = [i for i in unique_PIDs if i not in gotlist]
    return tbd

tbd = get_tbd()

def wrapper(ids):
    if type(ids).__name__ == 'list':
        assert len(ids) < 6
    if type(ids).__name__ == 'str':
        ids = [ids]
    fd = copy.deepcopy(fdict)
    fd['PAT_ID']['vals'] = ids
    q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
    q += "where pe.ENTRY_TIME >= '2017-01-01'"
    out = get_from_clarity_then_save(q, clar_conn=clar_conn)
    out.to_pickle(f"{outdir}tmp/notes_{'_'.join(ids)}.pkl")



pool = mp.Pool(mp.cpu_count())
start = time.time()
pool.map(wrapper, tbd, chunksize=1)
print(time.time() - start)
pool.close()

# now combine them into a single data  frame, and then split-off a metadata frame,
# it should be smaller and simpler to work with
if "raw_notes_df.pkl" not in os.listdir(datadir):
    tbd = get_tbd()
    assert len(tbd) == 0
    already_got = [i for i in os.listdir(f"{outdir}/tmp/") if 'notes_' in i]
    def proc(path):
        try:
            df = pd.read_pickle(path)
            if df.shape != (0, 0):
                df = combine_notes_by_type(df, CSN = 'CSN', note_type="NOTE_TYPE")
                return df
        except Exception as e:
            message = f"ERROR AT {path}"
            return message

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    raw_notes_df = pool.map(proc, [outdir + "tmp/" + i for i in already_got], chunksize=1)
    print(time.time() - start)
    pool.close()

    errs = [i for i in raw_notes_df if type(i).__name__ == "str"]
    assert len(errs)==0
    raw_notes_df = pd.concat(raw_notes_df)
    raw_notes_df.to_pickle(f"{datadir}raw_notes_df.pkl")

    # now pull out transplants -- figure out who had a transplant and lose any encounters from before the transplant
    unique_PIDs = raw_notes_df.PAT_ID.unique().tolist()
    unique_PIDs.sort()
    base_query = open("_8_transplant_query.sql").read()
    fdict = dict(PAT_ID={"vals": [], "foreign_table":"ti",
                              "foreign_key":"PAT_ID"}
                 )

    def wrapper(ids):
        if type(ids).__name__ == 'str':
            ids = [ids]
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        q+="\nwhere tc.TX_CLASS_C = 2 and ti.TX_SURG_DT is not NULL"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        return out

    chunks = [i * 1000 for i in list(range(len(unique_PIDs)//1000+1))]
    chunkids = [unique_PIDs[i:(i+1000)] for i in chunks]
    pool = mp.Pool(1)
    start = time.time()
    txplists = pool.map(wrapper, chunkids, chunksize=1)
    print(time.time() - start)
    pool.close()
    txpdf = pd.concat(txplists)
    txpdf.shape

    raw_notes_df = raw_notes_df.merge(txpdf, how = "left")
    raw_notes_df = raw_notes_df[(raw_notes_df.TX_SURG_DT > raw_notes_df.ENC_DATE) | (raw_notes_df.TX_SURG_DT.isnull())]

    # Keep the ones with at least three visits within the past 12 months
    # so, first sort by MRN and then by date.  then you can proceed linearly.
    raw_notes_df = raw_notes_df.sort_values(["PAT_ID", "ENC_DATE"])
    # two conditions:
    same_patient_conditon = (raw_notes_df.PAT_ID.shift(periods=2) == raw_notes_df.PAT_ID).astype(int)
    time_conditon = ((raw_notes_df.ENC_DATE - raw_notes_df.ENC_DATE.shift(periods=2)). \
                     dt.total_seconds() / 60 / 60 / 24 / 365 < 1).astype(int)
    # unify them
    condition = same_patient_conditon * time_conditon
    print(raw_notes_df.PAT_ID[condition.astype(bool)].nunique())
    print(raw_notes_df.PAT_ID.nunique())
    raw_notes_df = raw_notes_df[raw_notes_df.PAT_ID.isin(raw_notes_df.PAT_ID[condition.astype(bool)].unique())]

    # pull the dx associated with each CSN
    base_query = open("_8_diagnosis_query_all.sql").read()
    fdict = dict(CSN={"vals": [], "foreign_table": "dx",
                      "foreign_key": "CSN"}
                 )
    def wrapper(ids):
        fd = copy.deepcopy(fdict)
        fd['CSN']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        l = []
        for i in out.CSN.unique().tolist():
            l.append(dict(CSN=i, dxs=','.join(out.icd10[out.CSN == i].unique().tolist())))
        return pd.DataFrame(l)


    csnlist = raw_notes_df.CSN.unique().astype(int).tolist()
    chunks = [i * 1000 for i in list(range(len(csnlist) // 1000 + 1))]
    chunkids = [csnlist[i:(i + 1000)] for i in chunks]
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    dxdflist = pool.map(wrapper, chunkids, chunksize=1)
    print(time.time() - start)
    pool.close()

    dxdf = pd.concat(dxdflist)
    raw_notes_df = raw_notes_df.merge(dxdf, how='left')
    raw_notes_df.drop(columns="TX_SURG_DT", inplace=True)

    raw_notes_df.to_pickle(f"{outdir}raw_notes_df.pkl")
else:
    raw_notes_df = pd.read_pickle(f"{outdir}raw_notes_df.pkl")






#
# # do a batch download.  first specify a base query, and iteratively substitute out the ":ids" string with
# # blobs of UIDs.  as above, it only does this if it's not already on disk.
# if "op_encounters_df.json.bz2" in os.listdir(datadir): # this is the final output
#     op_encounters_df = pd.read_json("{0}op_encounters_df.json.bz2".format(datadir))
# else: # this gets the intermediate output from clarity
#     unique_PIDs = diagnosis_df.PAT_ID.unique().tolist()
#     unique_PIDs.sort()
#     base_query = open("_8_patient_query.sql").read()
#     res = []
#     batchsize = 5000
#     i = 0
#     elapsed = 0
#     while i < len(unique_PIDs):
#         start_time = time.time()
#         batch_string = "('" + "','".join(unique_PIDs[i:(i + batchsize)]) + "')"
#         batch_query = re.sub(":ids", batch_string, base_query)
#         batch_output = get_from_clarity_then_save(
#             query = batch_query,
#             clar_conn = clar_conn,
#             save_path=None
#         )
#         i+=batchsize
#         res.append(pd.DataFrame(batch_output))
#         elapsed += time.time() - start_time
#         print(str(i) + " of " + str(len(unique_PIDs)) + " after " + str(elapsed) + ". Got " + str(batch_output.shape))
#     op_encounters_df = pd.concat(res, ignore_index=True)
#     # Keep the ones with at least three visits within the past 12 months
#     # so, first sort by MRN and then by date.  then you can proceed linearly.
#     op_encounters_df = op_encounters_df.sort_values(["PAT_ID", "ENTRY_TIME"])
#     # two conditions:
#     same_patient_conditon = (op_encounters_df.PAT_ID.shift(periods=2) == op_encounters_df.PAT_ID).astype(int)
#     time_conditon = ((op_encounters_df.ENTRY_TIME - op_encounters_df.ENTRY_TIME.shift(periods=2)). \
#                      dt.total_seconds() / 60 / 60 / 24 / 365 < 1).astype(int)
#     # unify them
#     condition = same_patient_conditon * time_conditon
#     print(str(sum(condition)) + " patient encounters sort in, given our criteria")
#     op_encounters_df = op_encounters_df.loc[condition == 1]
#     # save it
#     op_encounters_df.to_json("{0}op_encounters_df.json.bz2".format(datadir))
#
#
# # get the transplant cases.  you'll filter the combined notes based on them, below
# txp_query = open("_8_transplant_query.sql").read()
# txp_df = get_from_clarity_then_save(
#     query=txp_query,
#     clar_conn=clar_conn,
#     save_path=f'{datadir}txp_df.json.bz2'
# )
# unique_pat_id = combined_notes_df.PAT_ID.unique()
#
# # Now get the notes.  They'll be batched again.
# if "combined_notes_df.json.bz2" in os.listdir(datadir): # this is the final output
#     combined_notes_df = pd.read_json("{0}combined_notes_df.json.bz2".format(datadir))
# elif "notes_df.json.bz2" in os.listdir(datadir):
#     notes_df = pd.read_json("{0}notes_df.json.bz2".format(datadir))
#     notes_df['PAT_ENC_CSN_ID'] = notes_df['PAT_ENC_CSN_ID'].astype(int)
#     combined_notes_df = combine_all_notes(notes_df, op_encounters_df)
#     combined_notes_df.to_json("{0}combined_notes_df.json.bz2".format(datadir))
#     del notes_df
# else:
#     base_query = open("_8_notes_query.sql").read()
#     unique_encounters = op_encounters_df["PAT_ENC_CSN_ID"].astype(str).tolist()
#     unique_encounters.sort()
#     res = []
#     batchsize = 500
#     i = 0
#     elapsed = 0
#     while i < len(unique_encounters):
#         start_time = time.time()
#         batch_string = "('" + "','".join(unique_encounters[i:(i + batchsize)]) + "')"
#         batch_query = re.sub(":ids", batch_string, base_query)
#         batch_output = get_from_clarity_then_save(
#             query=batch_query,
#             clar_conn=clar_conn,
#             save_path=None
#         )
#         i += batchsize
#         res.append(pd.DataFrame(batch_output))
#         elapsed += time.time() - start_time
#         print(str(i) + " of " + str(len(unique_encounters)) + " after " + str(elapsed)
#               + ".  Size of this batch was " + str(batch_output.shape)
#               + ".  It uses " + str(sys.getsizeof(batch_output)/1e6) + " mb of space.")
#     notes_df = pd.concat(res, ignore_index=True)
#     notes_df.to_json("{0}notes_df.json.bz2".format(datadir))
#     del res
#     notes_df['PAT_ENC_CSN_ID'] = notes_df['PAT_ENC_CSN_ID'].astype(int)
#     combined_notes_df = combine_all_notes(notes_df, op_encounters_df)
#     # now exclude the transplant cases
#     # merge the txp df by patent ID
#     merged = combined_notes_df[["PAT_ENC_CSN_ID", "PAT_ID", "ENTRY_TIME"]].merge(txp_df, on="PAT_ID", how="inner")
#     # now go through and lose all CSNs that happened after the surgery
#     CSNs_to_drop = merged[merged.ENTRY_TIME < merged.TX_SURG_DT].PAT_ENC_CSN_ID
#     combined_notes_df = combined_notes_df[~combined_notes_df.PAT_ENC_CSN_ID.isin(CSNs_to_drop)]
#     combined_notes_df.to_json("{0}combined_notes_df.json.bz2".format(datadir))
#     del notes_df
#
# print("We're getting " + str(combined_notes_df.shape[0]) + " notes from " \
#       + str(len(combined_notes_df.PAT_ID.unique())) + " people")
#
# # Next, get all diagnoses for all of these patients that we've got.
# # The result should be an array with MRN and DX as a text string, that can be merged onto any of the data frames that
# # we've got.
# if "all_dx_df.json.bz2" in os.listdir(datadir): # this is the final output
#     all_dx_df = pd.read_json("{0}all_dx_df.json.bz2".format(datadir))
# else:
#     base_query = open("_8_diagnosis_query_all_by_ID.sql").read()
#     unique_PIDs = combined_notes_df["PAT_ID"].astype(str).tolist()
#     unique_PIDs.sort()
#     res = []
#     batchsize = 5000
#     i = 0
#     elapsed = 0
#     while i < len(unique_PIDs):
#         start_time = time.time()
#         batch_string = "('" + "','".join(unique_PIDs[i:(i + batchsize)]) + "')"
#         batch_query = re.sub(":ids", batch_string, base_query)
#         batch_output = get_from_clarity_then_save(
#             query=batch_query,
#             clar_conn=clar_conn,
#             save_path=None
#         )
#         i += batchsize
#         res.append(pd.DataFrame(batch_output))
#         elapsed += time.time() - start_time
#         print(str(i) + " of " + str(len(unique_PIDs)) + " after " + str(elapsed)
#               + ".  Size of this batch was " + str(batch_output.shape)
#               + ".  It uses " + str(sys.getsizeof(batch_output) / 1e6) + " mb of space.")
#     all_dx_df = pd.concat(res, ignore_index=True)
#     all_dx_df.to_json("{0}all_dx_df.json.bz2".format(datadir))
#     del res
#
#
#
#
# # Finally, separate the notes into a CSV that is just notes and CSNs, and another that has all of the metadata.
# # this should allow me to filter on the metadata, while calling only the parts of the notes taht I need at a particular
# # moment
# znotes = combined_notes_df.drop(labels="NOTE_TEXT", axis = 1)
# znotes.to_json(f"{datadir}znotes.json.bz2")
#
# combined_notes_df[["PAT_ENC_CSN_ID", "NOTE_TEXT"]].to_csv(f"{datadir}note_text.csv")
#
# # save other files as CSVs, which are more convenient sometimes...
# znotes.to_csv(f"{datadir}znotes.csv")
# all_dx_df.to_csv(f"{datadir}dx_df.csv")
