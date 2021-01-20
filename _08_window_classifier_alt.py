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


def slidingWindow(sequence, winSize, step=1):
    # Verify the inputs
    # winSize is the total size of the window (including the center word)
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception(
            "**ERROR** winSize must not be larger than sequence length.")
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - (winSize - 1)) / step)
    # Start half a window into the text
    # Create a window by adding words on either side of the center word
    for i in range(int(np.ceil((winSize - 1) / 2)) + 1,
                   (int(np.floor((winSize - 1) / 2)) + numOfChunks + 1) * step,
                   step):
        yield sequence[i - (int(np.ceil((winSize - 1) / 2)) + 1): i + int(
            np.floor((winSize - 1) / 2))]


# get experiment number from command line arguments
assert len(sys.argv) == 2, 'Exp number must be specified as an argument'
exp = sys.argv[1]
exp = f"exp{exp}_lin_trees_WIN"

# get the correct directories
dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/",
        "/media/drv2/andrewcd2/frailty/output/",
        "/share/gwlab/frailty/output/"]
for d in dirs:
    if os.path.exists(d):
        datadir = d
outdir = f"{datadir}lin_trees_WIN/"
SVDdir = f"{outdir}svd/"
embeddingsdir = f"{outdir}embeddings/"
trtedatadir = f"{outdir}trtedata/"
sheepish_mkdir(outdir)
sheepish_mkdir(SVDdir)
sheepish_mkdir(embeddingsdir)
sheepish_mkdir(trtedatadir)

# load the notes from 2018
notes_2018 = [i for i in os.listdir(datadir + "notes_labeled_embedded/") if
              int(i.split("_")[-2][1:]) < 13]

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
df = pd.concat([pd.read_csv(datadir + "notes_labeled_embedded/" + i) for i in
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

# make a sliding 11-token "window" of tokens to classify
count = 0
window3 = None
win_size = 11
for i in list(df2.note.unique()):
    # Create sliding window of 11 tokens for each note
    note_i = df2.loc[df2['note'] == i]
    chunks = slidingWindow(note_i['token'].tolist(), winSize=win_size)
    # now concatenate output from generator separated by blank space
    window = []
    for each in chunks:
        window.append(' '.join(each))
    # repeat the first and final windows (first and last few tokens
    # ((winSize-1)/2) will have off-center windows)
    window2 = list(np.repeat(window[0], np.floor((win_size - 1) / 2)))
    window2.extend(window)
    repeats_end = list(
        np.repeat(window2[len(window2) - 1], np.ceil((win_size - 1) / 2)))
    window2.extend(repeats_end)
    # repeat for all notes
    if window3 is None:
        window3 = window2
    else:
        window3.extend(window2)
    count += 1
# add windows to df
df2['window'] = window3

# label windows with heirarchical rule
# at each token, find the max for each dummy for an 11-token window
window_label = None
for i in list(df_dums.note.unique()):
    # important to copy the data in this step to avoid chained indexing
    note_i = df_dums.loc[df_dums['note'] == i].copy()
    for v in list(y_dums.columns):
        # rolling max
        note_i[f"any_{v}"] = note_i[f"{v}"].rolling(window=11, center=True,
                                                    min_periods=0).max()
    if window_label is None:
        window_label = note_i
    else:
        window_label = window_label.append(note_i, ignore_index=True)
# apply pos/neg/neutral label using heirarchical rule
# 0 = neg, 1 = pos, 2 = neut
for n in out_varnames:
    window_label[f"{n}_pos"] = np.where(
        (window_label[f"any_{n}_1"] == 1), 1, 0)
    window_label[f"{n}_neg"] = np.where(
        ((window_label[f"{n}_pos"] != 1) &
         (window_label[f"any_{n}_-1"] == 1)), 1,
        0)
    window_label[f"{n}_neut"] = np.where(
        ((window_label[f"{n}_pos"] != 1) &
         (window_label[f"{n}_neg"] != 1)), 1, 0)
#keep labels only
window_label = window_label.iloc[:, -12:]


# split off embeddings
embeddings = df2.loc[:,
             df2.columns.str.startswith('note') | df2.columns.str.startswith(
                 'identity')].copy()
# for each note, calculate the mean of the embeddings for each 11 token rolling window. Get embeddings for the words before (lag) and after (lead) the center word.
count = 0
embeddings2 = None
for i in list(embeddings.note.unique()):
    # important to copy the data in this step to avoid chained indexing
    note_i = embeddings.loc[embeddings['note'] == i].copy()
    for v in range(0, (
    note_i.loc[:, note_i.columns.str.startswith('identity_')].shape[1])):
        # rolling min, max, mean
        note_i[f"min_{v}"] = note_i[f"identity_{v}"].rolling(window=11,
                                                             center=True,
                                                             min_periods=0).min()
        note_i[f"max_{v}"] = note_i[f"identity_{v}"].rolling(window=11,
                                                             center=True,
                                                             min_periods=0).max()
        note_i[f"mean_{v}"] = note_i[f"identity_{v}"].rolling(window=11,
                                                              center=True,
                                                              min_periods=0).mean()
    if embeddings2 is None:
        embeddings2 = note_i
    else:
        embeddings2 = embeddings2.append(note_i, ignore_index=True)
    count += 1

# drop embeddings for center word
embeddings2 = embeddings2.loc[:,
              ~embeddings2.columns.str.startswith('identity')].copy()

# drop embeddings from df2
df2 = df2.loc[:, ~df2.columns.str.startswith('identity')].copy()
#add labels
df2 = pd.concat([df2, window_label], axis=1)

# split into 10 folds, each containing different notes
notes = list(df2.note.unique())
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
    f_tr = df2[~df2.note.isin(fold)]
    f_te = df2[df2.note.isin(fold)]
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
        nnweight = 1 / np.mean(non_neutral[~df2.note.isin(fold)])
        caseweights = np.ones(df2.shape[0])
        caseweights[non_neutral.astype(bool)] *= nnweight
        tr_caseweights = caseweights[~df2.note.isin(fold)]
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
        f"{embeddingsdir}f_{f + 1}_tr_embed_min_max_mean_WIN.csv")
    embeddings_te.to_csv(
        f"{embeddingsdir}f_{f + 1}_te_embed_min_max_mean_WIN.csv")
    f_tr_svd300.to_csv(f"{SVDdir}f_{f + 1}_tr_svd300.csv")
    f_tr_svd1000.to_csv(f"{SVDdir}f_{f + 1}_tr_svd1000.csv")
    f_te_svd300.to_csv(f"{SVDdir}f_{f + 1}_te_svd300.csv")
    f_te_svd1000.to_csv(f"{SVDdir}f_{f + 1}_te_svd1000.csv")

end = timer()
duration = end - start
f = open(f"{trtedatadir}duration_WIN.txt", "w")
f.write(str(duration))
f.close()
