import os
from copy import deepcopy
import re
import pandas as pd
import numpy as np
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

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

# split into training and validation
np.random.seed(2670095)
trnotes = np.random.choice(notes_2018, len(notes_2018) * 2 // 3, replace=False)
tenotes = [i for i in notes_2018 if i not in trnotes]
trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

# define some useful constants
str_varnames = df.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df.columns if re.match("identity", i)]
out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# dummies for the outcomes
y_dums = pd.concat([pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
df = pd.concat([y_dums, df], axis=1)

# get a vector of non-negatives for case weights
tr_cw = []
for v in out_varnames:
    non_neutral = np.array(np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype \
        ('float32')
    nnweight = 1 / np.mean(non_neutral[df.note.isin(trnotes)])
    caseweights = np.ones(df.shape[0])
    caseweights[non_neutral.astype(bool)] *= nnweight
    tr_caseweights = caseweights[df.note.isin(trnotes)]
    tr_cw.append(tr_caseweights)


# Start by resetting the index
df = df.reset_index()
# Drop embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')]


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


#training & tests sets
tr_df = df2.loc[df.note.isin(trnotes),:]
te_df = df2.loc[df.note.isin(tenotes),:]


# Convert text into matrix of tf-idf features:
# id documents
tr_docs = tr_df['window'].tolist()
# instantiate countvectorizer (turn off default stopwords)
cv = CountVectorizer(analyzer='word', stop_words=None)
# compute tf
tr_df_tf = cv.fit_transform(tr_docs)
    # check shape (215097 windows and 20270 tokens)
    tr_df_tf.shape
    len(tr_df)
    # print first 10 docs again
    print(tr_df.iloc[0:10])
    # print count matrix for first 10 windows to visualize
    df_tf = pd.DataFrame(tr_df_tf.toarray(), columns=cv.get_feature_names())
    df_tf.loc[1:10, :]
# id additional stopwords: medlist_was_here_but_got_cut, meds_was_here_but_got_cut, catv2_was_here_but_got_cut
cuttext = '_was_here_but_got_cut'
stopw = [i for i in list(cv.get_feature_names()) if re.search(cuttext, i)]
#import ntlk stopwords and add stopw
from nltk.corpus import stopwords
en_stopwords = stopwords.words('english')
en_stopwords.extend(stopw)
    len(en_stopwords)
cv = CountVectorizer(analyzer='word', stop_words=en_stopwords)
# fit to data, then transform to count matrix
tr_df_tf = cv.fit_transform(tr_docs)
# fit to count matrix, then transform to tf-idf representation
tfidf_transformer = TfidfTransformer()
tr_df_tfidf = tfidf_transformer.fit_transform(tr_df_tf)
    # sort and print idf
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf"])
    df_idf.sort_values(by=['idf'])
    # print example
    df_tf_idf = pd.DataFrame(tr_df_tfidf[0].T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
    df_tf_idf.sort_values(by=['tfidf'], ascending=False)
    #compare to window
    tr_docs[0]
    #visualize another way
    df_tf_idf2 = pd.DataFrame(tr_df_tfidf[0].todense(), columns=cv.get_feature_names())
    #print sparse matrix for window 1
    print(df_tf_idf2)


# apply feature extraction to test set
# making sliding window
te_docs = te_df['window'].tolist()
te_df_tf = cv.transform(te_docs)
te_df_tfidf = tfidf_transformer.transform(te_df_tf)
    # check work:
    print(te_df.iloc[0:10].loc[:, ['token', 'window']])
    print(te_df.iloc[len(te_df) - 10:len(te_df)].loc[:, ['token', 'window']])
    # print example
    df_tf_idf = pd.DataFrame(te_df_tfidf[0].T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
    df_tf_idf.sort_values(by=['tfidf'], ascending=False)
    #compare to window
    te_docs[0]
    #visualize another way
    df_tf_idf2 = pd.DataFrame(te_df_tfidf[0].todense(), columns=cv.get_feature_names())
    #print sparse matrix for window 1
    print(df_tf_idf2)


## Output for r
scipy.io.mmwrite(f"{outdir}/tr_df_tfidf.mtx", tr_df_tfidf)
tr_df.to_csv(f"{outdir}/tr_df.csv")
scipy.io.mmwrite(f"{outdir}/te_df_tfidf.mtx", te_df_tfidf)
te_df.to_csv(f"{outdir}/te_df.csv")



