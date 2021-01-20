'''
Same as _08_sentence_classifier.py, except it makes a much smaller data set
where each fold contains 1 compressed note
'''


import os
import random
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer


pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

# get the correct directories
dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/",
        "/media/drv2/andrewcd2/frailty/output/",
        "/share/gwlab/frailty/output/"]
for d in dirs:
    if os.path.exists(d):
        datadir = d
if datadir == dirs[0]:  # mb
    notesdir = datadir
if datadir == dirs[1]:  # grace
    notesdir = f"{os.getcwd()}/output/"
if datadir == dirs[2]:  # azure
    notesdir = datadir
outdir = f"{datadir}lin_trees_TEST/"
SVDdir = f"{outdir}svd/"
embeddingsdir = f"{outdir}embeddings/"
trtedatadir = f"{outdir}trtedata/"
sheepish_mkdir(outdir)
sheepish_mkdir(SVDdir)
sheepish_mkdir(embeddingsdir)
sheepish_mkdir(trtedatadir)

# load SENTENCES
# check for .csv in filename to avoid the .DSstore file
# load the notes from 2018
notes_2018 = [i for i in
              os.listdir(notesdir + "notes_labeled_embedded_SENTENCES/")
              if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{datadir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
        cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
# generate 'note' label (used in webanno and notes_labeled_embedded)
uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
# conc_notes_df contains official list of eligible patients
notes_2018_in_cndf = [i for i in notes_2018 if
                      "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if
                  "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
# get notes_labeled_embedded that match eligible patients only
df = pd.concat(
    [pd.read_csv(notesdir + "notes_labeled_embedded_SENTENCES/" + i,
    dtype={'sent_start': 'boolean', 'token': 'string', 'PAT_ID': 'string'}) for i in
     notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)
# reset the index
df2 = df.reset_index(drop=True)

# set seed
seed = 111120

# define some useful constants
str_varnames = ['n_encs', 'n_ed_visits', 'n_admissions', 'days_hospitalized',
 'mean_sys_bp', 'mean_dia_bp', 'sd_sys_bp', 'sd_dia_bp', 'bmi_mean',
 'bmi_slope', 'max_o2', 'spo2_worst', 'ALBUMIN', 'ALKALINE_PHOSPHATASE', 'AST',
 'BILIRUBIN', 'BUN', 'CALCIUM', 'CO2', 'CREATININE', 'HEMATOCRIT', 'HEMOGLOBIN',
 'LDL', 'MCHC', 'MCV', 'PLATELETS', 'POTASSIUM', 'PROTEIN', 'RDW', 'SODIUM', 'WBC',
 'sd_ALBUMIN', 'sd_ALKALINE_PHOSPHATASE', 'sd_AST', 'sd_BILIRUBIN', 'sd_BUN',
 'sd_CALCIUM', 'sd_CO2', 'sd_CREATININE', 'sd_HEMATOCRIT', 'sd_HEMOGLOBIN',
 'sd_LDL', 'sd_MCHC', 'sd_MCV', 'sd_PLATELETS', 'sd_POTASSIUM', 'sd_PROTEIN',
 'sd_RDW', 'sd_SODIUM', 'sd_WBC', 'n_ALBUMIN', 'n_ALKALINE_PHOSPHATASE',
 'n_AST', 'n_BILIRUBIN', 'n_BUN', 'n_CALCIUM', 'n_CO2', 'n_CREATININE',
 'n_HEMATOCRIT', 'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
 'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC', 'FERRITIN', 'IRON',
 'MAGNESIUM', 'TRANSFERRIN', 'TRANSFERRIN_SAT', 'sd_FERRITIN', 'sd_IRON',
 'sd_MAGNESIUM', 'sd_TRANSFERRIN', 'sd_TRANSFERRIN_SAT', 'n_FERRITIN',
 'n_IRON', 'n_MAGNESIUM', 'n_TRANSFERRIN', 'n_TRANSFERRIN_SAT', 'PT', 'sd_PT',
 'n_PT', 'PHOSPHATE', 'sd_PHOSPHATE', 'n_PHOSPHATE', 'PTT', 'sd_PTT', 'n_PTT',
 'TSH', 'sd_TSH', 'n_TSH', 'n_unique_meds', 'elixhauser', 'n_comorb', 'AGE',
 'SEXFemale', 'SEXMale', 'MARITAL_STATUSMarried', 'MARITAL_STATUSOther',
 'MARITAL_STATUSSingle', 'MARITAL_STATUSWidowed', 'EMPY_STATFull.Time',
 'EMPY_STATNot.Employed', 'EMPY_STATOther', 'EMPY_STATPart.Time',
 'EMPY_STATRetired', 'MV_n_encs', 'MV_n_ed_visits', 'MV_n_admissions',
 'MV_days_hospitalized', 'MV_mean_sys_bp', 'MV_mean_dia_bp', 'MV_sd_sys_bp',
 'MV_sd_dia_bp', 'MV_bmi_mean', 'MV_bmi_slope', 'MV_max_o2', 'MV_spo2_worst',
 'MV_ALBUMIN', 'MV_ALKALINE_PHOSPHATASE', 'MV_AST', 'MV_BILIRUBIN', 'MV_BUN',
 'MV_CALCIUM', 'MV_CO2', 'MV_CREATININE', 'MV_HEMATOCRIT', 'MV_HEMOGLOBIN',
 'MV_LDL', 'MV_MCHC', 'MV_MCV', 'MV_PLATELETS', 'MV_POTASSIUM', 'MV_PROTEIN',
 'MV_RDW', 'MV_SODIUM', 'MV_WBC', 'MV_sd_ALBUMIN', 'MV_sd_ALKALINE_PHOSPHATASE',
 'MV_sd_AST', 'MV_sd_BILIRUBIN', 'MV_sd_BUN', 'MV_sd_CALCIUM', 'MV_sd_CO2',
 'MV_sd_CREATININE', 'MV_sd_HEMATOCRIT', 'MV_sd_HEMOGLOBIN', 'MV_sd_LDL',
 'MV_sd_MCHC', 'MV_sd_MCV', 'MV_sd_PLATELETS', 'MV_sd_POTASSIUM',
 'MV_sd_PROTEIN', 'MV_sd_RDW', 'MV_sd_SODIUM', 'MV_sd_WBC', 'MV_n_ALBUMIN',
 'MV_n_ALKALINE_PHOSPHATASE', 'MV_n_AST', 'MV_n_BILIRUBIN', 'MV_n_BUN',
 'MV_n_CALCIUM', 'MV_n_CO2', 'MV_n_CREATININE', 'MV_n_HEMATOCRIT',
 'MV_n_HEMOGLOBIN', 'MV_n_LDL', 'MV_n_MCHC', 'MV_n_MCV', 'MV_n_PLATELETS',
 'MV_n_POTASSIUM', 'MV_n_PROTEIN', 'MV_n_RDW', 'MV_n_SODIUM', 'MV_n_WBC',
 'MV_FERRITIN', 'MV_IRON', 'MV_MAGNESIUM', 'MV_TRANSFERRIN',
 'MV_TRANSFERRIN_SAT', 'MV_sd_FERRITIN', 'MV_sd_IRON', 'MV_sd_MAGNESIUM',
 'MV_sd_TRANSFERRIN', 'MV_sd_TRANSFERRIN_SAT', 'MV_n_FERRITIN', 'MV_n_IRON',
 'MV_n_MAGNESIUM', 'MV_n_TRANSFERRIN', 'MV_n_TRANSFERRIN_SAT', 'MV_PT',
 'MV_sd_PT', 'MV_n_PT', 'MV_PHOSPHATE', 'MV_sd_PHOSPHATE', 'MV_n_PHOSPHATE',
 'MV_PTT', 'MV_sd_PTT', 'MV_n_PTT', 'MV_TSH', 'MV_sd_TSH', 'MV_n_TSH',
 'MV_n_unique_meds', 'MV_n_comorb', 'MV_AGE', 'MV_SEX', 'MV_MARITAL_STATUS',
 'MV_EMPY_STAT']
embedding_colnames = [i for i in df2.columns if re.match("identity", i)]
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# set a unique sentence id that does not reset to 0 with each note
sentence = []
sent = -1
for s in range(df2.shape[0]):
    if df2.iloc[s]['sentence'] != df2.iloc[s - 1]['sentence']:
        sent += 1
    sentence.append(sent)
df2['sentence_id'] = sentence
# rename the note-specific sentence label
df2.rename(columns={'sentence': 'sentence_in_note'}, inplace=True)

# dummies for labels
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
y_dums = pd.concat(
    [pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
df2 = pd.concat([y_dums, df2], axis=1)

# label each sentence using heirachical rule:
# Positive label if any token is positive
# Negative label if there are no positive tokens and any token is negative
df2_label = df2.groupby('sentence_id', as_index=True).agg(
    note=('note', 'first'),
    sentence=('token', lambda x: ' '.join(x.astype(str))),  # sentence tokens
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
#on azure, I need to set the group as the index, then reset it (elsewhere, I
#can set 'as_index=False' and pandas will retain the group)
df2_label = df2_label.reset_index(drop=False)
# add negative & neutral label using heirarchical rule
for n in out_varnames:
    df2_label[f"{n}_neg"] = np.where(
        ((df2_label[f"{n}_pos"] != 1) & (df2_label[f"any_{n}_neg"] == 1)), 1,
        0)
    df2_label[f"{n}_neut"] = np.where(
        ((df2_label[f"{n}_pos"] != 1) & (df2_label[f"{n}_neg"] != 1)), 1, 0)
# drop extra columns
df2_label = df2_label.loc[:, ~df2_label.columns.str.startswith('any_')].copy()

# restrict to 10 notes that each contain all classes of all frailty aspects
note_labels = df2_label.groupby('note', as_index=True).agg(
    Msk_prob_pos=('Msk_prob_pos', max),
    Msk_prob_neg=('Msk_prob_neg', max),
    Msk_prob_neut=('Msk_prob_neut', max),
    Nutrition_pos=('Nutrition_pos', max),
    Nutrition_neg=('Nutrition_neg', max),
    Nutrition_neut=('Nutrition_neut', max),
    Resp_imp_pos=('Resp_imp_pos', max),
    Resp_imp_neg=('Resp_imp_neg', max),
    Resp_imp_neut=('Resp_imp_neut', max),
    Fall_risk_pos=('Fall_risk_pos', max),
    Fall_risk_neg=('Fall_risk_neg', max),
    Fall_risk_neut=('Fall_risk_neut', max),
)
#need to set then reset index, same as above
note_labels = note_labels.reset_index(drop=False)
note_labels['colsum'] = note_labels.iloc[:, 1:len(note_labels.columns)].sum(axis=1)
#restrict test notes to the firts 10 that have at least 1 of each tag
test_notes = list(note_labels[note_labels.colsum > 11].note[0:10])
df2_label = df2_label[df2_label.note.isin(test_notes)]
#restrict to sentences that are enriched for pos & neg tags
df2_label = df2_label[(df2_label.Msk_prob_neg > 0) | (df2_label.Msk_prob_pos > 0) |
(df2_label.Nutrition_neg > 0) | (df2_label.Nutrition_pos > 0) |
(df2_label.Resp_imp_neg > 0) | (df2_label.Resp_imp_pos > 0) |
(df2_label.Fall_risk_neg > 0) | (df2_label.Fall_risk_pos > 0)].reset_index(drop=True)
#get the same sentences from df2
df2 = df2[df2.sentence_id.isin(df2_label.sentence_id)].reset_index(drop=True)

# summarize embeddings (element-wise min/max/mean) for each sentence
# first, make empty df
clmns = ['sentence_id', 'note']
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    clmns.append(f"min_{v}")
    clmns.append(f"max_{v}")
    clmns.append(f"mean_{v}")
embeddings = pd.DataFrame(0, index=range(df2.sentence_id.nunique()),
                          columns=clmns)
embeddings['sentence_id'] = list(df2.sentence_id.drop_duplicates())
embeddings['note'] = df2.groupby('sentence_id', as_index=False)['note'].agg('first')['note']
# for each sentence, find the element-wise min/max/mean for embeddings
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    embeddings[f"min_{v}"] = df2.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        min)[f"identity_{v}"]
    embeddings[f"max_{v}"] = df2.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        max)[f"identity_{v}"]
    embeddings[f"mean_{v}"] = df2.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
        'mean')[f"identity_{v}"]

# drop embeddings for center word
embeddings2 = embeddings.loc[:,
              ~embeddings.columns.str.startswith('identity')].copy()

# make df of structured data and labels
# drop embeddings, notes (duplicate column)
str_lab = df2.loc[:, ~df2.columns.str.startswith('identity') &
                     ~df2.columns.str.startswith('note') &
                     ~df2.columns.str.startswith('Frailty_nos') &
                     ~df2.columns.str.endswith('_0') &
                     ~df2.columns.str.endswith('_1') &
                     ~df2.columns.str.endswith('_-1') &
                     ~df2.columns.isin(out_varnames)].copy()
# get one row of structured data for each sentence
str_lab = str_lab.groupby('sentence_id', as_index=True).first().reset_index(drop=False)
#check that sentence_ids match
assert sum(str_lab.sentence_id == df2_label.sentence_id) == len(
    str_lab), 'sentence_ids do not match'
# add labels & drop duplicate column
str_lab = pd.concat([str_lab, df2_label.drop(columns=['sentence_id'])], axis=1).copy()

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
        # Scale structured data
        scaler = StandardScaler()
        str_tr = scaler.fit_transform(f_tr[str_varnames])
        str_te = scaler.transform(f_te[str_varnames])
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
