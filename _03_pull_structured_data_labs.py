

'''
This script pulls structured data for patient windows corresponding to what is in 'conc_notes_df', generated
in the previous script.

Labs:  via the labs query.  Here's the list:
Creatinine, BUN (urea nitrogen), sodium, potassium, calcium, magnesium, phosphate (phosphorus), TSH, Aspartate aminotransferase (AST), alkaline phosphatase, total protein (protein, total), albumin, total bilirubin (bilirubin, total), Hemoglobin concentration (hemoglobin), hematocrit, MCV, MCHC, red blood cell distribution width (RDW), platelet concentration (platelets), white blood cells, Prothrombin time (PT), partial thromboplastin time (PTT), Mean systolic BP, mean diastolic BP, Hemoglobin A1c, carbon dioxide, iron, ferritin, transferrin, transferrin sat, LDL cholesterol

'''
import os
os.chdir('/Users/crandrew/projects/GW_PAIR_frailty_classifier/')
import pandas as pd
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, combine_notes_by_type, \
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
labs
'''
if "labs_raw.pkl" not in os.listdir(outdir):
    # load the file with all of the lab names
    labnames = read_txt("labnames_for_structured_data.txt").split('\n')

    bq = '''
    select
            ores.PAT_ENC_CSN_ID
        ,   pe.PAT_ID
        ,   ores.ORD_VALUE
        ,   ores.REFERENCE_UNIT
        ,   ores.RESULT_TIME
        ,   cc.COMMON_NAME
    from ORDER_RESULTS as ores
    join CLARITY_COMPONENT as cc on ores.COMPONENT_ID = cc.COMPONENT_ID
    join PAT_ENC as pe on pe.PAT_ENC_CSN_ID = ores.PAT_ENC_CSN_ID
    '''
    fdict = dict(PAT_ID={"vals": [], "foreign_table":"pe",
                              "foreign_key":"PAT_ID"},
                 COMMON_NAME={"vals": labnames, "foreign_table":"cc",
                              "foreign_key":"COMMON_NAME"}
                 )
    def wrapper(ids):
        start = time.time()
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
        q += "where ores.RESULT_TIME >= '2017-01-01'"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        print(f"did file starting with {ids[0]}, and it's got {out.shape}.  it took {time.time()-start}")
        return out


    UIDs = df.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    labout = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    labdf = pd.concat(labout)
    labdf.to_pickle(f"{outdir}labs_raw.pkl")

    # aggregate the labs and quality check them
    labdf.loc[labdf.COMMON_NAME == "UREA NITROGEN", "COMMON_NAME"] = "BUN"
    labdf.loc[labdf.COMMON_NAME == "POTASSIUM PLASMA", "COMMON_NAME"] = "POTASSIUM"
    labdf.loc[labdf.COMMON_NAME == "PHOSPHOROUS", "COMMON_NAME"] = "PHOSPHATE"
    labdf.loc[labdf.COMMON_NAME == "PHOSPHORUS", "COMMON_NAME"] = "PHOSPHATE"
    labdf.loc[labdf.COMMON_NAME == "TSH W/REFLEX TO FT4", "COMMON_NAME"] = "TSH"
    labdf.loc[labdf.COMMON_NAME == "TSH+ FREE T4", "COMMON_NAME"] = "TSH"
    labdf.loc[labdf.COMMON_NAME == "TSH, 3RD GENERATION (REFL)", "COMMON_NAME"] = "TSH"
    labdf.loc[labdf.COMMON_NAME == "TSH-ICMA", "COMMON_NAME"] = "TSH"
    labdf.loc[labdf.COMMON_NAME == "HYPOTHYROIDISM/TSH", "COMMON_NAME"] = "TSH"
    labdf.loc[labdf.COMMON_NAME == "ALKALINE PHOSPHATASE", "COMMON_NAME"] = "ALKALINE_PHOSPHATASE"
    labdf.loc[labdf.COMMON_NAME == "ALK PHOS, TOTAL", "COMMON_NAME"] = "ALKALINE_PHOSPHATASE"
    labdf.loc[labdf.COMMON_NAME == "PROTEIN TOTAL", "COMMON_NAME"] = "PROTEIN"
    labdf.loc[labdf.COMMON_NAME == "TOTAL PROTEIN", "COMMON_NAME"] = "PROTEIN"
    labdf.loc[labdf.COMMON_NAME == "BILIRUBIN TOTAL", "COMMON_NAME"] = "BILIRUBIN"
    labdf.loc[labdf.COMMON_NAME == "MEAN CELLULAR VOLUME", "COMMON_NAME"] = "MCV"
    labdf.loc[labdf.COMMON_NAME == "MEAN CELLULAR HEMOGLOBIN CONCENTRATION", "COMMON_NAME"] = "MCHC"
    labdf.loc[labdf.COMMON_NAME == "BICARBONATE", "COMMON_NAME"] = "CO2"
    labdf.loc[labdf.COMMON_NAME == "CARBON DIOXIDE", "COMMON_NAME"] = "CO2"
    labdf.loc[labdf.COMMON_NAME == "TOTAL IRON BINDING C", "COMMON_NAME"] = "TRANSFERRIN"
    labdf.loc[labdf.COMMON_NAME == "TOTAL IRON BINDING CAPACITY", "COMMON_NAME"] = "TRANSFERRIN"
    labdf.loc[labdf.COMMON_NAME == "IRON BINDING", "COMMON_NAME"] = "TRANSFERRIN"
    labdf.loc[labdf.COMMON_NAME == "IRON BINDING CAPACITY", "COMMON_NAME"] = "TRANSFERRIN"
    labdf.loc[labdf.COMMON_NAME == "TRANSFERRIN SATURATI", "COMMON_NAME"] = "TRANSFERRIN_SAT"
    labdf.loc[labdf.COMMON_NAME == "TRANSFERRIN SATURATION", "COMMON_NAME"] = "TRANSFERRIN_SAT"
    labdf.loc[labdf.COMMON_NAME == "IRON SATURATION", "COMMON_NAME"] = "TRANSFERRIN_SAT"
    labdf.loc[labdf.COMMON_NAME == "CHOLESTEROL CALCULATED LOW DENSITY LIPOPROTEIN", "COMMON_NAME"] = "LDL"
    labdf.loc[labdf.COMMON_NAME == "CHOLESTEROL DIRECT LDL", "COMMON_NAME"] = "LDL"
    labdf.loc[labdf.COMMON_NAME == "LOW DENSITY LIPOPROT", "COMMON_NAME"] = "LDL"
    labdf.loc[labdf.COMMON_NAME == "QLDL", "COMMON_NAME"] = "LDL"
    labdf.loc[labdf.COMMON_NAME == "Q LEUKOCYTES", "COMMON_NAME"] = "WBC"
    

    # convert  to numeric
    labdf['VAL_NUM'] = np.nan
    # use the limits where < and > are specified
    labdf.ORD_VALUE = labdf.ORD_VALUE.str.replace(">", "")
    labdf.ORD_VALUE = labdf.ORD_VALUE.str.replace("<", "")


    # convert to float where possible, throw an error otherwise
    def f(x):
        try:
            return float(x)
        except Exception:
            return -999.0


    labdf['VAL_NUM'] = labdf.ORD_VALUE.apply(f)
    # drop the ones that don't fit the filter
    labdf = labdf.loc[labdf.VAL_NUM != -999.0]
    labdf = labdf.drop(columns="ORD_VALUE")

    # names = labdf.COMMON_NAME.unique()
    # names.sort()
    # print(names)
    # # go through the variables, one at a time, and fix their units where needed
    # i = 0
    #
    # x = labdf.REFERENCE_UNIT.loc[labdf.COMMON_NAME == names[i]].unique()
    # print(names[i])
    # print(x)
    # print(labdf.loc[(labdf.COMMON_NAME == names[i])].groupby('REFERENCE_UNIT').mean())
    # print(labdf.loc[(labdf.COMMON_NAME == names[i])].groupby('REFERENCE_UNIT').count())
    # i += 1

    # convert mg/dl albumin to grams
    labdf.loc[(labdf.REFERENCE_UNIT == "mg/dL") & (labdf.COMMON_NAME == "ALBUMIN"), "VAL_NUM"] /= 1000
    labdf.loc[(labdf.REFERENCE_UNIT == "mg/dL") & (labdf.COMMON_NAME == "ALBUMIN"), "REFERENCE_UNIT"] = 'g/dL'
    # drop the percent in Alkaline phosphotase
    labdf = labdf.drop(labdf[(labdf.COMMON_NAME == "ALKALINE_PHOSPHATASE") & (labdf.REFERENCE_UNIT == "%")].index)
    print(labdf.shape)
    # drop hemoglobin when no in g/dl
    labdf = labdf.drop(labdf[(labdf.COMMON_NAME == "HEMOGLOBIN") & (labdf.REFERENCE_UNIT != "g/dL")].index)
    print(labdf.shape)
    # drop LDL with nanomoles
    labdf = labdf.drop(labdf[(labdf.COMMON_NAME == "LDL") & (labdf.REFERENCE_UNIT == "nmol/L")].index)
    print(labdf.shape)
    # drop protein when not in g/dl
    labdf = labdf.drop(labdf[(labdf.COMMON_NAME == "HEMOGLOBIN") & (labdf.REFERENCE_UNIT == "mg/dL")].index)
    # drop the reference unit field now that they're all standardized
    labdf = labdf.drop(columns = "REFERENCE_UNIT")
    # save
    labdf.to_pickle(f"{outdir}labs_raw.pkl")

else:
    labdf = pd.read_pickle(f"{outdir}labs_raw.pkl")


'''
Now, build a dataset that pulls their means, SDs, and counts
'''
# pull the most recent CSN from the concatenated notes df
df['PAT_ENC_CSN_ID'] = df.CSNS.apply(lambda x: int(x.split(",")[0]))
# write a function that takes a row of the concatenated notes DF and outputs a dict of lab values
def recent_labs(i):
    try:
        ldf = labdf.loc[(labdf.PAT_ID == df.PAT_ID.iloc[i]) &
                        (labdf.RESULT_TIME <= df.LATEST_TIME.iloc[i] + pd.DateOffset(days=1)) & # the deals with the fact that the times might be rounded down to the nearest day sometimes
                        (labdf.RESULT_TIME >= df.LATEST_TIME.iloc[i] - pd.DateOffset(months=6))
                        ]
        if nrow(ldf) > 0:
            g = ldf.groupby(["COMMON_NAME"])
            mu = g['VAL_NUM'].mean()
            sig = g['VAL_NUM'].std()
            N = g['VAL_NUM'].count()
            outdict = {**{**dict(zip(mu.index.tolist(), mu.tolist())),
                          **dict(zip(["sd_"+ j for j in sig.index.tolist()], sig.tolist()))},
                        **dict(zip(["n_"+ j for j in N.index.tolist()], N.tolist()))}
            return(outdict)
        else:
            return
    except Exception as e:
        print(f"======={i}======")
        print(e)
        return e
        
pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
rlabs = pool.map(recent_labs, range(nrow(df)), chunksize=1)
print(time.time() - start)
pool.close()

# oh snap!  I forgot to put the coordinates in
assert(len(rlabs) == nrow(df))
for i in range(len(rlabs)):
    if type(rlabs[i]).__name__ == "dict":
        rlabs[i].update({'PAT_ID':df.PAT_ID.iloc[i]})
        rlabs[i].update({'PAT_ENC_CSN_ID':df.PAT_ENC_CSN_ID.iloc[i]})
    


errs = [i for i in range(len(rlabs)) if type(rlabs[i]).__name__ != "dict"]
rlabs_fixed = [i for i in rlabs if type(i).__name__ == "dict"]

labs_6m = pd.DataFrame(rlabs_fixed)


labs_6m.to_pickle(f"{outdir}labs_6m.pkl")



#
# '''
# Labs: there are many, many names for similar labs.
# Pull all labs for the patients in our sample, and use this to map all of the synonymous lab names to common names
# '''
# fdict = dict(PAT_ID={"vals": [], "foreign_table":"pe",
#                           "foreign_key":"PAT_ID"})
#
#
# def wrapper(ids):
#     fd = copy.deepcopy(fdict)
#     fd['PAT_ID']['vals'] = ids
#     q = query_filtered_with_temp_tables(bq, fd, rstring=str(np.random.choice(10000000000)))
#     q += "where pe.ENTRY_TIME >= '2017-01-01'"
#     out = get_from_clarity_then_save(q, clar_conn=clar_conn)
#     return out
#
# UIDs = df.PAT_ID.unique().tolist()
# UIDs.sort()
# chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]
#
#
# pool = mp.Pool(processes=mp.cpu_count())
# start = time.time()
# labnames = pool.map(wrapper, chunks, chunksize=1)
# print(time.time() - start)
# pool.close()
#
# labs = pd.concat(labnames)
# labs = labs.sort_values("COMMON_NAME")
# pd.Series(labs.COMMON_NAME.unique().tolist()).to_csv(f'{outdir}/labnames.csv')
#
#
