
'''
Takes the output of _05_ingest_to_sentence.py and does the following:
1.) Concatenate notes that are eligible for training/validation
2.) Label each sentence using the "heirarchical rule"
3.) For each sentence, get the labels, structured data, and element-wise
 min/max/mean for the embeddings.
4.) Split the data into 10 folds for cross validation
5.) For each fold, get the output labels, PCA of structured data, caseweights,
embeddings and 300- & 1000-d SVD of tf-idf
'''

import os
import random
import re
from configargparse import ArgParser
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from _99_project_module import send_message_to_slack

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

# batchstring = 'testJ21'

def main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
    options = p.parse_args()
    zipfile = options.zipfile
    batchstring = options.batchstring
    # identify the active learning directory
    outdir = f"{os.getcwd()}/output/"
    datadir = f"{os.getcwd()}/data/"
    ALdir = f"{outdir}/saved_models/AL{batchstring}"
    sheepish_mkdir(ALdir)
    # make files for the CSV output
    sheepish_mkdir(f"{ALdir}/processed_data/")
    sheepish_mkdir(f"{ALdir}/processed_data/svd")
    sheepish_mkdir(f"{ALdir}/processed_data/embeddings")
    sheepish_mkdir(f"{ALdir}/processed_data/trtedata")
    notes_2018 = [i for i in
              os.listdir(outdir + "notes_labeled_embedded_SENTENCES/")
              if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    # generate 'note' label (used in webanno and notes_labeled_embedded)
    cndf.month = cndf.month.astype(str)
    uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
    # conc_notes_df contains official list of eligible patients
    notes_2018_in_cndf = [i for i in notes_2018 if
                          "_".join(i.split("_")[-2:]) in uidstr]
    notes_excluded = [i for i in notes_2018 if
                      "_".join(i.split("_")[-2:]) not in uidstr]
    assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
    # remove double-annotated notes
    '''
    25 Jan 2021:  removing double-annotated notes, and notes from different 
    span of same patient.  Move the actual files so they don't get picked up later.
    As such, the code below will only do culling once.
    It will also move notes that were excluded in the July 2020 cull
    '''
    pids = set([re.sub(".csv", "", i.split("_")[-1]) for i in notes_2018_in_cndf])
    keepers = []
    for i in pids:
        notes_i = [j for j in notes_2018_in_cndf if i in j]
        if len(notes_i) >1:
            # check and see if there are notes from different batches.  Use the last one if so
            batchstrings = [k.split("_")[1] for k in notes_i]
            assert all(["AL" in k for k in batchstrings]), "not all double-coded notes are from an AL round."
            in_latest_batch = [k for k in notes_i if max(batchstrings) in k]
            # deal with a couple of manual cases
            if i == '004081006':
                keepers.append('enote_AL01_v2_m2_004081006.csv')
            elif i == '044286789':
                keepers.append('enote_AL01_m2_044286789.csv')
            elif len(in_latest_batch) == 1:
                keepers.append(in_latest_batch[0])
            elif len(set([k.split("_")[-2] for k in in_latest_batch]))>1: # deal with different spans
                spans = [k.split("_")[-2] for k in in_latest_batch]
                latest_span = [k for k in in_latest_batch if max(spans) in k]
                assert len(latest_span) == 1
                keepers.append(latest_span[0])
            elif any(['v2' in k for k in in_latest_batch]): # deal with the case of the "v2" notes -- an outgrowth of confusion around the culling in July 2020
                v2_over_v1 = [k for k in in_latest_batch if 'v2' in k]
                assert len(v2_over_v1) == 1
                keepers.append(v2_over_v1[0])
            else:
                print('problem with culling')
                breakpoint()
        else:
            keepers.append(notes_i[0])
    assert len(keepers) == len(pids)
    droppers = [i for i in notes_2018_in_cndf if i not in keepers]
    if any(droppers):
        msg = "The following notes are getting moved/dropped because of duplication:" + "\n".join(droppers) + f"\n\nthere are {len(droppers)} of them"
        send_message_to_slack(msg)
    enote_dir = f"{outdir}/notes_labeled_embedded_SENTENCES"
    drop_dir = f"{enote_dir}/dropped_notes"
    sheepish_mkdir(drop_dir)
    assert os.path.exists(enote_dir)
    assert os.path.exists(drop_dir)
    for i in droppers:
        fra = enote_dir + "/" + i
        till = drop_dir + "/" + i
        os.rename(fra, till)
    assert all([i in os.listdir(enote_dir) for i in keepers])
    len(keepers)
    





# cruft starts here:
    # clean up the notes_2018_in_cndf
    len(notes_2018_in_cndf)
    
    [i for i in keepers if '004081006' in i]
    [i for i in notes_2018_in_cndf if '004081006' in i]
    
    jm = [ 'enote_AL00_m11_057800369.csv', 'enote_AL00_m4_057800369.csv', 'enote_AL00_m9_058589987.csv', 'enote_AL01_v2_m8_004081006.csv', 'enote_AL00_m8_004081006.csv', 'enote_AL00_m1_052080793.csv', 'enote_AL01_v2_m2_044286789.csv', 'enote_AL01_m3_Z3095842.csv', 'enote_AL00_m9_Z3095842.csv', 'enote_AL00_m12_054002217.csv', 'enote_AL00_m11_Z1816871.csv', 'enote_AL00_m1_Z2198505.csv', 'enote_AL00_m2_006364095.csv']
    [i for i in droppers if i not in jm]    
    [i for i in droppers if '004081006' in i]    


044286789    
    [i for i in keepers if '004081006' in i]
    [i for i in notes_2018_in_cndf if '044286789' in i]
    
    jm = [ 'enote_AL00_m11_057800369.csv', 'enote_AL00_m4_057800369.csv', 'enote_AL00_m9_058589987.csv', 'enote_AL01_v2_m8_004081006.csv', 'enote_AL00_m8_004081006.csv', 'enote_AL00_m1_052080793.csv', 'enote_AL01_v2_m2_044286789.csv', 'enote_AL01_m3_Z3095842.csv', 'enote_AL00_m9_Z3095842.csv', 'enote_AL00_m12_054002217.csv', 'enote_AL00_m11_Z1816871.csv', 'enote_AL00_m1_Z2198505.csv', 'enote_AL00_m2_006364095.csv']
    [i for i in droppers if i not in jm]    
    [i for i in droppers if '004081006' in i]    

[i for i in jm if '044286789' in i]
    
    df = pd.concat([pd.read_csv(outdir + "notes_labeled_embedded_SENTENCES/" + i,
                                dtype={'token': str, 'PAT_ID': str}) for i in
                    notes_2018_in_cndf])
    
    cc = 0
    xxx = {}
    for n, i in enumerate(notes_2018_in_cndf):
        xx = pd.read_csv(outdir + "notes_labeled_embedded_SENTENCES/" + i,
                                dtype={'token': str, 'PAT_ID': str})
        print(xx.shape)
        xxx[i] = xx
        print(i)
    
    [i for i in range(len(xxx)) if xxx[i].shape[0]==808]



pids = {}
# find the set of pat ids that are duplicated
for i in notes_2018_in_cndf:
    pid = re.sub(".csv", "", "_".join(i.split("_")[-2:]))
    try:
        pids[pid] += 1
    except:
        pids[pid] = 1

x2=0
x3=0
for i in pids:
    if pids[i] == 2:
        x2 +=1
    elif pids[i] == 3:
        x3 +=1


pids = {}
# find the set of pat ids that are duplicated
for i in notes_2018_in_cndf:
    pid = re.sub(".csv", "", i.split("_")[-1])
    try:
        pids[pid] += 1
    except:
        pids[pid] = 1

reps = []
for k in pids

for i in pids

(pd.DataFrame(pids, index = [0]) == 2).sum().sum()


for i in pids:
    if pids[i] >1:
        print(f"{[j for j in notes_2018_in_cndf if i in j]}")



[i for i in AL00 if 'Z3095842' in i]
[i for i in AL01 if 'Z3095842' in i]

[i for i in AL00 if '057800369' in i]
[i for i in AL01 if '057800369' in i]

[i for i in AL00 if '006364095' in i]
[i for i in AL01 if '006364095' in i]

[i for i in AL00 if '004081006' in i]
[i for i in AL01 if '004081006' in i]



AL00 = []
AL01 = []
batch = []
for i in notes_2018_in_cndf:
    if "AL00" in i:
        AL00.append(i)
    elif "AL01" in i:
        AL01.append(i)
    else:
        batch.append(i)
        
xxx['enote_AL01_m9_Z3095842.csv'].loc[:,out_varnames].mean()
xxx['enote_AL00_m9_Z3095842.csv'].loc[:,out_varnames].mean()
        



[i for i in notes_2018_in_cndf if "044286789" in i]
        
len(batch)
len(AL00)
len(AL01)
32+15+26

xxx[36].iloc[:10, :20]
xxx[38].iloc[:10, :20]
xxx[61].iloc[:10, :20]
xxx[38].iloc[:5, :5]

dd = xxx[38].loc[:,out_varnames] - xxx[36].loc[:,out_varnames]
dd.mean().mean()

xxx[38].note.unique()
xxx[36].note.unique()
xxx[36].loc[:,out_varnames].mean()



notes_2018_in_cndf[38]
notes_2018_in_cndf[36]
    df = df.drop(columns = ['Unnamed: 0', 'sent_start', 'length'])
    
    [i for i in notes_2018_in_cndf if re.sub(".csv", "", re.sub('enote_', "", i)) not in list(df.note.unique())]
    
    len(set(notes_2018_in_cndf))
    
    ###########
    # Load and process structured data
    strdat = pd.read_csv(f"{outdir}structured_data_merged_cleaned.csv", 
                         index_col = 0)
    
    # set seed
    seed = 8675309
    str_varnames = ['n_encs', 'n_ed_visits', 'n_admissions', 'days_hospitalized',
                    'mean_sys_bp', 'mean_dia_bp', 'sd_sys_bp', 'sd_dia_bp',
                    'bmi_mean', 'bmi_slope', 'max_o2', 'spo2_worst', 'ALBUMIN',
                    'ALKALINE_PHOSPHATASE', 'AST', 'BILIRUBIN', 'BUN', 'CALCIUM',
                    'CO2', 'CREATININE', 'HEMATOCRIT', 'HEMOGLOBIN', 'LDL', 'MCHC',
                    'MCV', 'PLATELETS', 'POTASSIUM', 'PROTEIN', 'RDW', 'SODIUM',
                    'WBC', 'sd_ALBUMIN', 'sd_ALKALINE_PHOSPHATASE', 'sd_AST',
                    'sd_BILIRUBIN', 'sd_BUN', 'sd_CALCIUM', 'sd_CO2',
                    'sd_CREATININE', 'sd_HEMATOCRIT', 'sd_HEMOGLOBIN', 'sd_LDL',
                    'sd_MCHC', 'sd_MCV', 'sd_PLATELETS', 'sd_POTASSIUM',
                    'sd_PROTEIN', 'sd_RDW', 'sd_SODIUM', 'sd_WBC', 'n_ALBUMIN',
                    'n_ALKALINE_PHOSPHATASE', 'n_AST', 'n_BILIRUBIN', 'n_BUN',
                    'n_CALCIUM', 'n_CO2', 'n_CREATININE', 'n_HEMATOCRIT',
                    'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
                    'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC',
                    'FERRITIN', 'IRON', 'MAGNESIUM', 'TRANSFERRIN',
                    'TRANSFERRIN_SAT', 'sd_FERRITIN', 'sd_IRON', 'sd_MAGNESIUM',
                    'sd_TRANSFERRIN', 'sd_TRANSFERRIN_SAT', 'n_FERRITIN', 'n_IRON',
                    'n_MAGNESIUM', 'n_TRANSFERRIN', 'n_TRANSFERRIN_SAT', 'PT',
                    'sd_PT', 'n_PT', 'PHOSPHATE', 'sd_PHOSPHATE', 'n_PHOSPHATE',
                    'PTT', 'sd_PTT', 'n_PTT', 'TSH', 'sd_TSH', 'n_TSH',
                    'n_unique_meds', 'elixhauser', 'n_comorb', 'AGE', 'SEX_Female',
                    'SEX_Male', 'MARITAL_STATUS_Divorced', 'MARITAL_STATUS_Married',
                    'MARITAL_STATUS_Other', 'MARITAL_STATUS_Single',
                    'MARITAL_STATUS_Widowed', 'EMPY_STAT_Disabled',
                    'EMPY_STAT_Full Time', 'EMPY_STAT_Not Employed',
                    'EMPY_STAT_Other', 'EMPY_STAT_Part Time', 'EMPY_STAT_Retired',
                    'MV_n_encs', 'MV_n_ed_visits', 'MV_n_admissions',
                    'MV_days_hospitalized', 'MV_mean_sys_bp', 'MV_mean_dia_bp',
                    'MV_sd_sys_bp', 'MV_sd_dia_bp', 'MV_bmi_mean', 'MV_bmi_slope',
                    'MV_max_o2', 'MV_spo2_worst', 'MV_ALBUMIN',
                    'MV_ALKALINE_PHOSPHATASE', 'MV_AST', 'MV_BILIRUBIN', 'MV_BUN',
                    'MV_CALCIUM', 'MV_CO2', 'MV_CREATININE', 'MV_HEMATOCRIT',
                    'MV_HEMOGLOBIN', 'MV_LDL', 'MV_MCHC', 'MV_MCV', 'MV_PLATELETS',
                    'MV_POTASSIUM', 'MV_PROTEIN', 'MV_RDW', 'MV_SODIUM', 'MV_WBC',
                    'MV_sd_ALBUMIN', 'MV_sd_ALKALINE_PHOSPHATASE', 'MV_sd_AST',
                    'MV_sd_BILIRUBIN', 'MV_sd_BUN', 'MV_sd_CALCIUM', 'MV_sd_CO2',
                    'MV_sd_CREATININE', 'MV_sd_HEMATOCRIT', 'MV_sd_HEMOGLOBIN',
                    'MV_sd_LDL', 'MV_sd_MCHC', 'MV_sd_MCV', 'MV_sd_PLATELETS',
                    'MV_sd_POTASSIUM', 'MV_sd_PROTEIN', 'MV_sd_RDW',
                    'MV_sd_SODIUM', 'MV_sd_WBC', 'MV_n_ALBUMIN',
                    'MV_n_ALKALINE_PHOSPHATASE', 'MV_n_AST', 'MV_n_BILIRUBIN',
                    'MV_n_BUN', 'MV_n_CALCIUM', 'MV_n_CO2', 'MV_n_CREATININE',
                    'MV_n_HEMATOCRIT', 'MV_n_HEMOGLOBIN', 'MV_n_LDL', 'MV_n_MCHC',
                    'MV_n_MCV', 'MV_n_PLATELETS', 'MV_n_POTASSIUM', 'MV_n_PROTEIN',
                    'MV_n_RDW', 'MV_n_SODIUM', 'MV_n_WBC', 'MV_FERRITIN',
                    'MV_IRON', 'MV_MAGNESIUM', 'MV_TRANSFERRIN',
                    'MV_TRANSFERRIN_SAT', 'MV_sd_FERRITIN', 'MV_sd_IRON',
                    'MV_sd_MAGNESIUM', 'MV_sd_TRANSFERRIN',
                    'MV_sd_TRANSFERRIN_SAT', 'MV_n_FERRITIN', 'MV_n_IRON',
                    'MV_n_MAGNESIUM', 'MV_n_TRANSFERRIN', 'MV_n_TRANSFERRIN_SAT',
                    'MV_PT', 'MV_sd_PT', 'MV_n_PT', 'MV_PHOSPHATE',
                    'MV_sd_PHOSPHATE', 'MV_n_PHOSPHATE', 'MV_PTT', 'MV_sd_PTT',
                    'MV_n_PTT', 'MV_TSH', 'MV_sd_TSH', 'MV_n_TSH',
                    'MV_n_unique_meds', 'MV_elixhauser', 'MV_n_comorb', 'MV_AGE',
                    'MV_SEX', 'MV_MARITAL_STATUS', 'MV_EMPY_STAT']
    out_varnames = ['Msk_prob', 'Nutrition', 'Resp_imp', 'Fall_risk']
    
    
    strdat = strdat.drop(columns = ['RACE', 'LANGUAGE', 'MV_RACE', 'MV_LANGUAGE'])
    strdat = strdat.merge(df[['PAT_ID','month']].drop_duplicates())
    strdat.shape
    
    
    df[['PAT_ID','month']].drop_duplicates().shape
    
    # drop the mean & sd for labs that are >70% missing. Keep the missing value
    # indicator (so rare labs become present/absent)
    missingness = df[str_varnames].isnull().sum()/df.shape[0]
    missingness = missingness.loc[missingness > 0.7].index
    str_varnames = np.setdiff1d(str_varnames, missingness)


df.columns

[i for i in str_varnames if i not in df.columns]

# set a unique sentence id that does not reset to 0 with each note
sentence = []
sent = -1
for s in range(df.shape[0]):
    if df.iloc[s]['sentence'] != df.iloc[s - 1]['sentence']:
        sent += 1
    sentence.append(sent)
df['sentence_id'] = sentence
# rename the note-specific sentence label
df.rename(columns={'sentence': 'sentence_in_note'}, inplace=True)

# dummies for labels
out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
y_dums = pd.concat(
    [pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
df = pd.concat([y_dums, df], axis=1)

# label each sentence using heirachical rule:
# Positive label if any token is positive
# Negative label if there are no positive tokens and any token is negative
df_label = df.groupby('sentence_id', as_index=True).agg(
    sentence=('token', lambda x: ' '.join(x.astype(str))),  # sentence tokens
    sentence_in_note=('sentence_in_note', 'first'),
    n_tokens=('token', 'count'),
    any_Msk_prob_neg=('Msk_prob_-1', max),
    Msk_prob_pos=('Msk_prob_1', max),
    any_Nutrition_neg=('Nutrition_-1', max),
    Nutrition_pos=('Nutrition_1', max),
    any_Resp_imp_neg=('Resp_imp_-1', max),
    Resp_imp_pos=('Resp_imp_1', max),
    any_Fall_risk_neg=('Fall_risk_-1', max),
    Fall_risk_pos=('Fall_risk_1', max),
)
# need to set the group as the index, then reset it (unclear why this is not
# equivalent to 'as_index=False')
df_label = df_label.reset_index(drop=False)
# add negative & neutral label using heirarchical rule
for n in out_varnames:
    df_label[f"{n}_neg"] = np.where(
        ((df_label[f"{n}_pos"] != 1) & (df_label[f"any_{n}_neg"] == 1)), 1,
        0)
    df_label[f"{n}_neut"] = np.where(
        ((df_label[f"{n}_pos"] != 1) & (df_label[f"{n}_neg"] != 1)), 1, 0)
# drop extra columns
df_label = df_label.loc[:, ~df_label.columns.str.startswith('any_')].copy()

# summarize embeddings (element-wise min/max/mean) for each sentence
# first, make empty df
clmns = ['sentence_id', 'note']
for v in range(0, df.columns.str.startswith('identity_').sum()):
    clmns.append(f"min_{v}")
    clmns.append(f"max_{v}")
    clmns.append(f"mean_{v}")
embeddings = pd.DataFrame(0, index=range(df.sentence_id.nunique()),
                          columns=clmns)
embeddings['sentence_id'] = list(df.sentence_id.drop_duplicates())
embeddings['note'] = df.groupby('sentence_id', as_index=False)['note'].agg('first')['note']
# for each sentence, find the element-wise min/max/mean for embeddings
for v in range(0, df.columns.str.startswith('identity_').sum()):
    embeddings[f"min_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        min)[f"identity_{v}"]
    embeddings[f"max_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        max)[f"identity_{v}"]
    embeddings[f"mean_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        'mean')[f"identity_{v}"]

# drop embeddings for center word
embeddings2 = embeddings.loc[:,
              ~embeddings.columns.str.startswith('identity')].copy()

# make df of structured data and labels
str_lab = df.loc[:, df.columns.isin(['PAT_ID', 'note', 'month',
                                           'sentence_id']) |
                     df.columns.isin(str_varnames)].copy()
# get one row of structured data for each sentence
str_lab = str_lab.groupby('sentence_id', as_index=True).first().reset_index(drop=False)
#check that sentence_ids match
assert sum(str_lab.sentence_id == df_label.sentence_id) == len(
    str_lab), 'sentence_ids do not match'
# add labels & drop duplicate column
str_lab = pd.concat([str_lab, df_label.drop(columns=['sentence_id'])], axis=1).copy()

##### REPEATED K-FOLD CROSS-VALIDATION #####
seed_start = 942020
# split into 10 folds, each containing different notes
notes = list(str_lab.note.unique())
# sort notes first in order to standardize the random split based on the seed
notes.sort()
# Set up repeats
for r in range(3):
    # shuffle differently for each repeat
    random.seed(seed_start + r)
    np.random.shuffle(notes)
    # make a list of notes in each of the 10 test folds
    fold_list = np.array_split(notes, 10)
    # Set up folds
    for f in range(10):
        # split fold
        fold = list(fold_list[f])
        # Identify training (k-1) folds and test fold
        f_tr = str_lab[~str_lab.note.isin(fold)].reset_index(drop=True)
        f_te = str_lab[str_lab.note.isin(fold)].reset_index(drop=True)
        # Impute structured data
        imp_mean = SimpleImputer(strategy='mean')
        str_tr = pd.DataFrame(imp_mean.fit_transform(f_tr[str_varnames]),
                              columns=f_tr[str_varnames].columns)
        str_te = pd.DataFrame(imp_mean.transform(f_te[str_varnames]),
                              columns=f_te[str_varnames].columns)
        # Scale structured data
        scaler = StandardScaler()
        str_tr = scaler.fit_transform(str_tr)
        str_te = scaler.transform(str_te)
        # Fit PCA on structured data and take 95% of variance, then scale again
        pca = PCA(n_components=0.95, svd_solver='full')
        str_tr = pd.DataFrame(scaler.fit_transform(pca.fit_transform(str_tr)))
        str_te = pd.DataFrame(scaler.transform(pca.transform(str_te)))
        pca_cols = ['pc_' + str(p) for p in str_tr.columns]
        str_tr.columns = pca_cols
        str_te.columns = pca_cols
        #replace structured data with PCA
        f_tr = pd.concat([f_tr[f_tr.columns[~f_tr.columns.isin(str_varnames)]],
                          str_tr], axis=1)
        f_te = pd.concat([f_te[f_te.columns[~f_te.columns.isin(str_varnames)]],
                          str_te], axis=1)
        # get embeddings for fold
        embeddings_tr = embeddings2[~embeddings2.note.isin(fold)].reset_index(drop=True)
        embeddings_te = embeddings2[embeddings2.note.isin(fold)].reset_index(drop=True)
        # test for matching length
        assert len(f_tr.note) == len(
            embeddings_tr.note), 'notes do not match embeddings'
        assert len(f_te.note) == len(
            embeddings_te.note), 'notes do not match embeddings'
        # get a vector of caseweights for each frailty aspect
        # weight non-neutral tokens by the inverse of their prevalence
        f_tr_cw = {}
        for v in out_varnames:
            non_neutral = np.array(np.sum(
                f_tr[[i for i in f_tr.columns if (v in i) and
                         (("_pos" in i) or ("_neg" in i))]], axis=1)).astype(
                'float32')
            nnweight = 1 / np.mean(non_neutral)
            tr_caseweights = np.ones(f_tr.shape[0])
            tr_caseweights[non_neutral.astype(bool)] *= nnweight
            f_tr_cw[f'{v}_cw'] = tr_caseweights
        # make cw df
        f_tr_cw = pd.DataFrame(f_tr_cw)
        # Convert text into matrix of tf-idf features:
        # id documents
        tr_docs = f_tr['sentence'].tolist()
        # instantiate countvectorizer (turn off default stopwords)
        cv = CountVectorizer(analyzer='word', stop_words=None)
        # compute tf
        f_tr_tf = cv.fit_transform(tr_docs)
        # id additional stopwords: medlist_was_here_but_got_cut,
        # meds_was_here_but_got_cut, catv2_was_here_but_got_cut
        cuttext = '_was_here_but_got_cut'
        stopw = [i for i in list(cv.get_feature_names()) if re.search(cuttext, i)]
        # repeat countvec with full list of stopwords
        cv = CountVectorizer(analyzer='word', stop_words=stopw)
        # fit to data, then transform to count matrix
        f_tr_tf = cv.fit_transform(tr_docs)
        # fit to count matrix, then transform to tf-idf representation
        tfidf_transformer = TfidfTransformer()
        f_tr_tfidf = tfidf_transformer.fit_transform(f_tr_tf)
        # apply feature extraction to test set (do NOT fit on test data)
        te_docs = f_te['sentence'].tolist()
        f_te_tf = cv.transform(te_docs)
        f_te_tfidf = tfidf_transformer.transform(f_te_tf)
        # dimensionality reduction with truncated SVD
        svd_300 = TruncatedSVD(n_components=300, n_iter=5, random_state=9082020)
        svd_1000 = TruncatedSVD(n_components=1000, n_iter=5, random_state=9082020)
        # fit to training data & transform
        f_tr_svd300 = pd.DataFrame(svd_300.fit_transform(f_tr_tfidf))
        f_tr_svd1000 = pd.DataFrame(svd_1000.fit_transform(f_tr_tfidf))
        # transform test data (do NOT fit on test data)
        f_te_svd300 = pd.DataFrame(svd_300.transform(f_te_tfidf))
        f_te_svd1000 = pd.DataFrame(svd_1000.transform(f_te_tfidf))
        # Output for r
        f_tr.to_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_tr_df.csv")
        f_te.to_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_te_df.csv")
        f_tr_cw.to_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_tr_cw.csv")
        embeddings_tr.to_csv(
            f"{embeddingsdir}r{r + 1}_f{f + 1}_tr_embed_min_max_mean_SENT.csv")
        embeddings_te.to_csv(
            f"{embeddingsdir}r{r + 1}_f{f + 1}_te_embed_min_max_mean_SENT.csv")
        f_tr_svd300.to_csv(f"{SVDdir}r{r + 1}_f{f + 1}_tr_svd300.csv")
        f_tr_svd1000.to_csv(f"{SVDdir}r{r + 1}_f{f + 1}_tr_svd1000.csv")
        f_te_svd300.to_csv(f"{SVDdir}r{r + 1}_f{f + 1}_te_svd300.csv")
        f_te_svd1000.to_csv(f"{SVDdir}r{r + 1}_f{f + 1}_te_svd1000.csv")





        
    # # get the correct directories
    # dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/notes_preprocessed_SENTENCES/",
    #         "/media/drv2/andrewcd2/frailty/output/notes_preprocessed_SENTENCES/",
    #         "/share/gwlab/frailty/output/notes_preprocessed_SENTENCES/",
    #         "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/notes_preprocessed_SENTENCES/"]
    # for d in dirs:
    #     if os.path.exists(d):
    #         notesdir = d
    # if datadir == dirs[0]:  # mb
    #     notesdir = datadir
    # if datadir == dirs[1]:  # grace
    #     notesdir = f"{os.getcwd()}/output/"
    # if datadir == dirs[2]:  # azure
    #     notesdir = datadir
    # outdir = f"{datadir}notes_preprocessed_SENTENCES/"
    # SVDdir = f"{outdir}svd/"
    # embeddingsdir = f"{outdir}embeddings/"
    # trtedatadir = f"{outdir}trtedata/"
    # sheepish_mkdir(outdir)
    # sheepish_mkdir(SVDdir)
    # sheepish_mkdir(embeddingsdir)
    # sheepish_mkdir(trtedatadir)

# load SENTENCES
# check for .csv in filename to avoid the .DSstore file
# load the notes from 2018

# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
# get notes_labeled_embedded that match eligible patients only
