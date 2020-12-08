import os
import random
import re
import sys
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


# test for missing values before training
def test_nan_inf(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nan.')
    if np.isinf(tensor).any():
        raise ValueError('Tensor contains inf.')


# get experiment number from command line arguments
assert len(sys.argv) == 2, 'Exp number must be specified as an argument'
exp = sys.argv[1]
exp = f"exp{exp}_lin_trees_SENT"

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
outdir = f"{datadir}lin_trees_SENT/"
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
notes_2018 = [i for i in os.listdir(notesdir + "notes_labeled_embedded_SENTENCES/")
              if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{datadir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
        cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
#generate 'note' label (used in webanno and notes_labeled_embedded)
uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
# conc_notes_df contains official list of eligible patients
notes_2018_in_cndf = [i for i in notes_2018 if
                      "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if
                  "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
# get notes_labeled_embedded that match eligible patients only
df = pd.concat([pd.read_csv(notesdir + "notes_labeled_embedded_SENTENCES/" + i) for i in
                notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)
# reset the index
df2 = df.reset_index()

# set seed
seed = 111120

# define some useful constants
str_varnames = df2.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df2.columns if re.match("identity", i)]
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# make dummies for the outcomes
y_dums = pd.concat(
    [pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
df_dums = pd.concat([y_dums, df2['note']], axis=1)

# set a unique sentence id
sentence = []
sent = -1
for s in range(df2.shape[0]):
    if df2.iloc[s]['sentence'] != df2.iloc[s - 1]['sentence']:
        sent += 1
    sentence.append(sent)
df2['sentence_id'] = sentence

#dummies for labels
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
y_dums = pd.concat(
    [pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
cols = list(['note', 'sentence', 'token']) + list(y_dums.columns)
df2 = pd.concat([y_dums, df2], axis=1)

#label each sentence using heirachical rule:
# Positive label if any token is positive
# Negative label if there are no positive tokens and any token is negative
df2_label = df2.groupby('sentence_id', as_index=False).agg(
    note=('note', 'first'),
    sentence=('token', lambda x: ' '.join(x.astype(str))), #sentence tokens
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
#add negative & neutral label using heirarchical rule
for n in out_varnames:
    df2_label[f"{n}_neg"] = np.where(((df2_label[f"{n}_pos"] != 1) & (df2_label[f"any_{n}_neg"] == 1)), 1, 0)
    df2_label[f"{n}_neut"] = np.where(((df2_label[f"{n}_pos"] != 1) & (df2_label[f"{n}_neg"] != 1)), 1, 0)
#drop extra columns
df2_label = df2_label.loc[:, ~df2_label.columns.str.startswith('any_')].copy()

# make empty df
clmns = ['sentence_id']
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    clmns.append(f"min_{v}")
    clmns.append(f"max_{v}")
    clmns.append(f"mean_{v}")
embeddings = pd.DataFrame(0, index=range(df2.sentence_id.nunique()), columns=clmns)
embeddings['sentence_id'] = list(df2.sentence_id.drop_duplicates())
# for each sentence, find the element-wise min/max/mean for embeddings
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    embeddings[f"min_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg(min)
    embeddings[f"max_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg(max)
    embeddings[f"mean_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg('mean')

# drop embeddings for center word
embeddings2 = embeddings.loc[:,
              ~embeddings.columns.str.startswith('identity')].copy()

# make df of structured data and labels
# drop embeddings
str_lab = df2.loc[:, ~df2.columns.str.startswith('identity') &
                     ~df2.columns.str.startswith('note')].copy()
# get one row of structured data for each sentence
str_lab = str_lab.groupby('sentence_id', as_index=False).first()
#add labels
str_lab = pd.concat([str_lab, df2_label], axis=1).copy()


# split into 10 folds, each containing different notes
notes = list(str_lab.note.unique())
# sort notes before randomly splitting in order to standardize the random split based on the seed
notes.sort()
random.seed(942020)
np.random.shuffle(notes)
# make a list of notes in each of the 10 test folds
fold_list = np.array_split(notes, 10)

# start timing tf-idf modeling strategy
start = timer()

##### CROSS-VALIDATION #####
# All steps past this point must be performed separately for each c-v fold
for f in range(10):
    # split fold
    fold = list(fold_list[f])
    # Identify training (k-1) folds and test fold
    f_tr = str_lab[~str_lab.note.isin(fold)]
    f_te = str_lab[str_lab.note.isin(fold)]
    # get embeddings for fold
    embeddings_tr = embeddings2[~embeddings2.note.isin(fold)]
    embeddings_te = embeddings2[embeddings2.note.isin(fold)]
    # test for matching length
    assert len(f_tr.note) == len(
        embeddings_tr.note), 'notes do not match embeddings'
    assert len(f_te.note) == len(
        embeddings_te.note), 'notes do not match embeddings'
    # get a vector of caseweights for each frailty aspect
    # weight non-neutral tokens by the inverse of their prevalence
    # e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
    f_tr_cw = {}
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if
                           ("_0" not in i) and (v in i)]], axis=1)).astype \
            ('float32')
        nnweight = 1 / np.mean(non_neutral[~str_lab.note.isin(fold)])
        caseweights = np.ones(str_lab.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[~str_lab.note.isin(fold)]
        f_tr_cw[f'{v}_cw'] = tr_caseweights
    # make cw df
    f_tr_cw = pd.DataFrame(f_tr_cw)
    # Convert text into matrix of tf-idf features:
    # id documents
    tr_docs = f_tr['window'].tolist()
    # instantiate countvectorizer (turn off default stopwords)
    cv = CountVectorizer(analyzer='word', stop_words=None)
    # compute tf
    f_tr_tf = cv.fit_transform(tr_docs)
    # id additional stopwords: medlist_was_here_but_got_cut, meds_was_here_but_got_cut, catv2_was_here_but_got_cut
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
    te_docs = f_te['window'].tolist()
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
    ## Output for r
    f_tr.to_csv(f"{trtedatadir}f_{f + 1}_tr_df.csv")
    f_te.to_csv(f"{trtedatadir}f_{f + 1}_te_df.csv")
    f_tr_cw.to_csv(f"{trtedatadir}f_{f + 1}_tr_cw.csv")
    embeddings_tr.to_csv(
        f"{embeddingsdir}f_{f + 1}_tr_embed_min_max_mean_SENT.csv")
    embeddings_te.to_csv(
        f"{embeddingsdir}f_{f + 1}_te_embed_min_max_mean_SENT.csv")
    f_tr_svd300.to_csv(f"{SVDdir}f_{f + 1}_tr_svd300.csv")
    f_tr_svd1000.to_csv(f"{SVDdir}f_{f + 1}_tr_svd1000.csv")
    f_te_svd300.to_csv(f"{SVDdir}f_{f + 1}_te_svd300.csv")
    f_te_svd1000.to_csv(f"{SVDdir}f_{f + 1}_te_svd1000.csv")

end = timer()
duration = end - start
f = open(f"{trtedatadir}duration_SENT.txt", "w")
f.write(str(duration))
f.close()

