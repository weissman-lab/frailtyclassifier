import os
import re
import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

#datadir = f"{os.getcwd()}/data/"
#outdir = f"{os.getcwd()}/output/"
datadir = "/media/drv2/andrewcd2/frailty/output/"
outdir = f"{os.getcwd()}/output/_08_window_classifier_alt/"

def slidingWindow(sequence, winSize, step=1):
    """Returns a generator that will iterate through
    the defined chunks of input sequence.  Input sequence
    must be iterable."""
    # Verify the inputs
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - winSize) / step)
    # Start half a window into the text
    # Center the window with 5 words on either side of the center word
    for i in range(int(winSize/2)+1, (int(winSize/2)+numOfChunks+1) * step, step):
        yield sequence[i - (int(winSize/2)+1) : i + int(winSize/2)]


# load the notes from 2018
notes_2018 = [i for i in os.listdir(datadir + "notes_labeled_embedded/") if int(i.split("_")[-2][1:]) < 13]

# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline 
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{datadir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
    cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
uidstr = ("m"+cndf.month.astype(str)+"_"+cndf.PAT_ID+".csv").tolist()

notes_2018_in_cndf = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)

df = pd.concat([pd.read_csv(datadir + "notes_labeled_embedded/" + i) for i in notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)

# define some useful constants
str_varnames = df.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)


# reset the index
df = df.reset_index()
# drop embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')].copy()

# dummies for the outcomes
y_dums = pd.concat([pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
df2 = pd.concat([y_dums, df2], axis=1)


# 1.) make new "documents," where each doc is an 11-token window (sliding window - 5 tokens on either side of center word)
#     Use the frailty label for the center word as the label for the document (11-token window)
# 2.) process the documents with scikit-learn tf-idf strategy
# note: windowing must be done on a note-by-note basis. Windows should not overlap two notes.
count = 0
window3 = None
for i in list(df2.note.unique()):
    #Create sliding window of 11 tokens for each note
    note_i = df2.loc[df2['note'] == i]
    chunks = slidingWindow(note_i['token'].tolist(), winSize=10)
    #now concatenate output from generator separated by blank space
    window = []
    for each in chunks:
        window.append(' '.join(each))
    #repeat the first and final windows (first 5 and last 5 tokens will have off-center windows)
    window2 = list(np.repeat(window[0], 5))
    window2.extend(window)
    repeats_end = list(np.repeat(window2[len(window2)-1], 5))
    window2.extend(repeats_end)
    #repeat for all notes
    if window3 is None:
        window3 = window2
    else:
        window3.extend(window2)
    count += 1
#add windows to df
df2['window'] = window3

#split into 10 folds, each containing different notes
#sort notes before randomly splitting in order to standardize the random split based on the seed
df2.sort_values('note')
notes=list(df2.note.unique())
random.seed(942020)
np.random.shuffle(notes)

##### CROSS-VALIDATION #####
# All steps past this point must be performed separately for each c-v fold
for f in range(9):
    #split fold
    fold = list(np.array_split(notes, 10)[f])
    # Identify training (k-1) folds and test fold
    f_tr = df2[~df2.note.isin(fold)]
    f_te = df2[df2.note.isin(fold)]

    # get a vector of caseweights for each frailty aspect
    # weight non-neutral tokens by the inverse of their prevalence
    # e.g. 1.3% of fall_risk tokens are non-neutral. Therefore, non-neutral tokens are weighted * (1/0.013)
    f_tr_cw = {}
    for v in out_varnames:
        non_neutral = np.array(
            np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype \
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
    svd_50 = TruncatedSVD(n_components=50, n_iter=5, random_state=9082020)
    svd_300 = TruncatedSVD(n_components=300, n_iter=5, random_state=9082020)
    svd_1000 = TruncatedSVD(n_components=1000, n_iter=5, random_state=9082020)
    # fit to training data & transform
    f_tr_svd50 = pd.DataFrame(svd_50.fit_transform(f_tr_tfidf))
    f_tr_svd300 = pd.DataFrame(svd_300.fit_transform(f_tr_tfidf))
    f_tr_svd1000 = pd.DataFrame(svd_1000.fit_transform(f_tr_tfidf))
    # transform test data (do NOT fit on test data)
    f_te_svd50 = pd.DataFrame(svd_50.transform(f_te_tfidf))
    f_te_svd300 = pd.DataFrame(svd_300.transform(f_te_tfidf))
    f_te_svd1000 = pd.DataFrame(svd_1000.transform(f_te_tfidf))

    ## Output for r
    f_tr.to_csv(f"{outdir}f_{f+1}_tr_df.csv")
    f_te.to_csv(f"{outdir}f_{f+1}_te_df.csv")
    f_tr_cw.to_csv(f"{outdir}f_{f+1}_tr_cw.csv")
    f_tr_svd50.to_csv(f"{outdir}f_{f+1}_tr_svd50.csv")
    f_tr_svd300.to_csv(f"{outdir}f_{f+1}_tr_svd300.csv")
    f_tr_svd1000.to_csv(f"{outdir}f_{f+1}_tr_svd1000.csv")
    f_te_svd50.to_csv(f"{outdir}f_{f+1}_te_svd50.csv")
    f_te_svd300.to_csv(f"{outdir}f_{f+1}_te_svd300.csv")
    f_te_svd1000.to_csv(f"{outdir}f_{f+1}_te_svd1000.csv")