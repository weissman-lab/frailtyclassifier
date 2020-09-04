import os
import re
import random
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"

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
notes_2018 = [i for i in os.listdir(outdir + "notes_labeled_embedded/") if int(i.split("_")[-2][1:]) < 13]

# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline 
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
    cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
uidstr = ("m"+cndf.month.astype(str)+"_"+cndf.PAT_ID+".csv").tolist()

notes_2018_in_cndf = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)

df = pd.concat([pd.read_csv(outdir + "notes_labeled_embedded/" + i) for i in notes_2018])
df.drop(columns='Unnamed: 0', inplace=True)


# define some useful constants
str_varnames = df.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# dummies for the outcomes
y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
df = pd.concat([y_dums, df], axis=1)


# Start by resetting the index
df = df.reset_index()
# Drop embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')].copy()


# 1.) make new "documents". Each "document" contains a 11-token
#     window (sliding window - 5 tokens on either side of center word)
#     Use the frailty label for the center word as the label for the document (11-token window)
# 2.) process the documents with scikit-learn tf-idf
#     strategy (capture Zach's MWE)
#windowing must be done on a note-by-note basis. Windows should not overlap two notes.
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
    #test
    print(count)
    len(window3)
    len(df2)
    print(df2.iloc[0:10].loc[:, ['token', 'window']])
    print(df2.iloc[len(df2) - 10:len(df2)].loc[:, ['token', 'window']])
    notetest = df2[df2.note == 'AL00_m7_049412554']
    print(notetest.iloc[0:10].loc[:, ['token', 'window']])
    print(notetest.iloc[len(notetest) - 10:len(notetest)].loc[:, ['token', 'window']])
    notetest.iloc[len(notetest)-1]['window']


#split into 10 folds, each containing different notes
notes=list(df2.note.unique())
random.seed(942020)
np.random.shuffle(notes)
fold1 = list(np.array_split(notes, 10)[0])
fold2 = list(np.array_split(notes, 10)[1])
fold3 = list(np.array_split(notes, 10)[2])
fold4 = list(np.array_split(notes, 10)[3])
fold5 = list(np.array_split(notes, 10)[4])
fold6 = list(np.array_split(notes, 10)[5])
fold7 = list(np.array_split(notes, 10)[6])
fold8 = list(np.array_split(notes, 10)[7])
fold9 = list(np.array_split(notes, 10)[8])
fold10 = list(np.array_split(notes, 10)[9])
#label test fold for each batch
#df['fold1'] = np.where(df.note.isin(fold1), 1, 0)
#df['fold2'] = np.where(df.note.isin(fold2), 1, 0)
#df['fold3'] = np.where(df.note.isin(fold3), 1, 0)
#df['fold4'] = np.where(df.note.isin(fold4), 1, 0)
#df['fold5'] = np.where(df.note.isin(fold5), 1, 0)
#df['fold6'] = np.where(df.note.isin(fold6), 1, 0)
#df['fold7'] = np.where(df.note.isin(fold7), 1, 0)
#df['fold8'] = np.where(df.note.isin(fold8), 1, 0)
#df['fold9'] = np.where(df.note.isin(fold9), 1, 0)
#df['fold10'] = np.where(df.note.isin(fold10), 1, 0)



#code everything for fold 1 (where fold1 = 1 is the hold out test set)
#tf-idf works for fold 1
#add case weights to the df?
#maybe truncatedSVD for feature selection? or just send to r for univariate logistic regression
#then write it as a loop over all of the folds
tr_f1 = df2[~df2.note.isin(fold1)]
te_f1 = df2[df2.note.isin(fold1)]


# get a vector of non-negatives for case weights
tr_cw = []
for v in out_varnames:
    non_neutral = np.array(np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype \
        ('float32')
    nnweight = 1 / np.mean(non_neutral[df2.note.isin(fold1)])
    caseweights = np.ones(df2.shape[0])
    caseweights[non_neutral.astype(bool)] *= nnweight
    tr_caseweights = caseweights[df2.note.isin(fold1)]
    tr_cw.append(tr_caseweights)


# Convert text into matrix of tf-idf features:
# id documents
tr_docs = tr_f1['window'].tolist()
# instantiate countvectorizer (turn off default stopwords)
cv = CountVectorizer(analyzer='word', stop_words=None)
# compute tf
tr_f1_tf = cv.fit_transform(tr_docs)
    # check shape (215097 windows and 20270 tokens)
    tr_f1_tf.shape
    len(tr_f1)
    # print first 10 docs again
    print(tr_f1.iloc[0:10])
    # print count matrix for first 10 windows to visualize
    df_tf = pd.DataFrame(tr_f1_tf.toarray(), columns=cv.get_feature_names())
    df_tf.loc[1:10, :]
# id additional stopwords: medlist_was_here_but_got_cut, meds_was_here_but_got_cut, catv2_was_here_but_got_cut
cuttext = '_was_here_but_got_cut'
stopw = [i for i in list(cv.get_feature_names()) if re.search(cuttext, i)]
cv = CountVectorizer(analyzer='word', stop_words=stopw)
# fit to data, then transform to count matrix
tr_f1_tf = cv.fit_transform(tr_docs)
# fit to count matrix, then transform to tf-idf representation
tfidf_transformer = TfidfTransformer()
tr_f1_tfidf = tfidf_transformer.fit_transform(tr_f1_tf)
    # sort and print idf
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf"])
    df_idf.sort_values(by=['idf'])
    # print example
    df_tf_idf = pd.DataFrame(tr_f1_tfidf[0].T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
    df_tf_idf.sort_values(by=['tfidf'], ascending=False)
    #compare to window
    tr_docs[0]
    #visualize another way
    df_tf_idf2 = pd.DataFrame(tr_f1_tfidf[0].todense(), columns=cv.get_feature_names())
    #print sparse matrix for window 1
    print(df_tf_idf2)


# apply feature extraction to test set
# making sliding window
te_docs = te_f1['window'].tolist()
te_f1_tf = cv.transform(te_docs)
te_f1_tfidf = tfidf_transformer.transform(te_f1_tf)
    # check work:
    print(te_f1.iloc[0:10].loc[:, ['token', 'window']])
    print(te_f1.iloc[len(te_f1) - 10:len(te_f1)].loc[:, ['token', 'window']])
    # print example
    df_tf_idf = pd.DataFrame(te_f1_tfidf[0].T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
    df_tf_idf.sort_values(by=['tfidf'], ascending=False)
    #compare to window
    te_docs[0]
    #visualize another way
    df_tf_idf2 = pd.DataFrame(te_f1_tfidf[0].todense(), columns=cv.get_feature_names())
    #print sparse matrix for window 1
    print(df_tf_idf2)


#add some kind of feature selection? or just export to R for univariate logistic regression




## Output for r
scipy.io.mmwrite(f"{outdir}/tr_df_tfidf.mtx", tr_df_tfidf)
tr_df.to_csv(f"{outdir}/tr_df.csv")
scipy.io.mmwrite(f"{outdir}/te_df_tfidf.mtx", te_df_tfidf)
te_df.to_csv(f"{outdir}/te_df.csv")



