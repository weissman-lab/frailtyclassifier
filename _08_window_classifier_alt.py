import os
from copy import deepcopy
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = f"{os.getcwd()}/data/"
outdir = f"{os.getcwd()}/output/"

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

# write_txt(",".join(["_".join(i.split("_")[-2:]) for i in notes_excluded]), f"{outdir}cull_list_15jul.txt")

df = pd.concat([pd.read_csv(outdir + "notes_labeled_embedded/" + i) for i in notes_2018])
df.drop(columns='Unnamed: 0', inplace=True)

# 1.) make new "documents". Each "document" contains a 11-token window (sliding window - 5 tokens on either side of center word)
# 2.) Use the frailty label for the center word as the label for the document (11-token window)
# 3.) process the documents with the standard scikit-learn tf-idf strategy (I think you can capture Zach's tokenization
# by manually setting the tokens to be separated by spaces, carriage returns, and punctuation)

# Making new documents
# Start by resetting the index
df2 = df.reset_index()
# Drop everything except index & token
df2 = df2.iloc[:, 0:2]
# Later, switch back to just dropping embeddings
# df2 = df.loc[:, ~df.columns.str.startswith('identity')]

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

#generator output
chunks = slidingWindow(df2['token'].tolist(),winSize=10)
#now take output from generator and concatenate into a 'document'
window = []
for each in chunks:
    window.append(' '.join(each))
#repeat the first and final windows (first 5 and last 5 tokens will have off-center windows)
repeats_start = list(np.repeat(window[0], 5))
repeats_start.extend(window)
window2=repeats_start
repeats_end = list(np.repeat(window2[len(window2)-1], 5))
window2.extend(repeats_end)
#add windows to df
df3 = deepcopy(df2)
df3['window'] = window2
    # check work:
    print(df3.iloc[0:10])
    print(df3.iloc[len(df3)-10:len(df3)])

# Convert text into matrix of tf-idf features:
# id documents
docs = df3['window'].tolist()
# instantiate countvectorizer with default behavior (lowercase=True, remove default stop words)
cv = CountVectorizer(analyzer='word')
# compute tf
tf = cv.fit_transform(docs)
    # check shape (215097 windows and 20270 tokens)
    tf.shape
    len(df3)
    # print first 10 docs again
    print(df3.iloc[0:10])
    # print count matrix for first 10 windows to visualize
    df_tf = pd.DataFrame(tf.toarray(), columns=cv.get_feature_names())
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
# compute tf
tf = cv.fit_transform(docs)
# compute idf
tfidf_transformer=TfidfTransformer()
tfidf_transformer.fit(tf)
    # sort and print idf
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=["idf"])
    df_idf.sort_values(by=['idf'])
# compute tfidf
tf_idf=tfidf_transformer.transform(tf)
    # print example
    df_tf_idf = pd.DataFrame(tf_idf[0].T.todense(), index=cv.get_feature_names(), columns=['tfidf'])
    df_tf_idf.sort_values(by=['tfidf'], ascending=False)
    #compare to window
    docs[0]
    #visualize another way
    df_tf_idf2 = pd.DataFrame(tf_idf[0].todense(), columns=cv.get_feature_names())
    #print sparse matrix for window 1
    print(df_tf_idf2)





# split into training and validation
np.random.seed(mainseed)
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


# models






cuttext = '(medlist_was_here_but_got_cut|meds_was_here_but_got_cut|catv2_was_here_but_got_cut)'

# create tokenizer to preserve original tokens (which are currently separated with whitespace)
# also, instate default tokenizing behavior (select tokens of 2 or more alphanumeric characters; punctuation is completely ignored and always treated as a token separator)
def preserve_token(text):
    return (w for w in re.split('\\s+',text)
            if re.match('(?u)\b\w\w+\b', w))

return (w for w in re.match('(?u)\b\w\w+\b', df2['token'].tolist()))

tokens = preserve_token(docs)
toklist = []
for each in tokens:
    toklist.append(each)

#
def preserve_token(text):
    return re.split("\\s+",text)

# instantiate countvectorizer with our tokenizer and other default behavior (lowercase=True, remove default stop words)
cv = CountVectorizer(tokenizer=preserve_token,
                     lowercase=True,
                     analyzer='word')

stopword


cuttext = 'medlist_was_here_but_got_cut'
full_list =
test_list = ['medlist_was_here_but_got_cut_1', 'medlist_was_here_but_got_cut_2']
res = [x for x in test_list if re.search(cuttext, x)]
# initializing list
test_list = ['GeeksforGeeks', 'Geeky', 'Computers', 'Algorithms']
# initializing substring
subs = 'Geek'
# using re + search()
# to get string with substring
res = [x for x in test_list if re.search(subs, x)]
# printing result
print("All strings with given substring are : " + str(res))


















print(tf_idf[0])

#print counts as an array
print(tf.toarray())
#print tokens
print(cv.get_feature_names())
#print count matrix
df_tf = pd.DataFrame(tf.toarray(),
                     columns=cv.get_feature_names())
df_tf.loc[1:10, ['progress', 'notes']]


# fix windowing (center it on the center word; will need to chop off hte first 5 or so rows first then add them back on + repeats like i did at the end
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

sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
chunks = slidingWindow(sequence,winSize=10)
for each in chunks:
    print(each)

sequence = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']
winSize = 10
step = 1
numOfChunks = int((len(sequence) - winSize) / step)
range(int(winSize/2)+1, (int(winSize/2)+numOfChunks) * step, step)
i = 10
sequence[i - (int(winSize/2)+1) : i + int(winSize/2)]
# 0 1 2 3 4 5 6 7 8  9 10 11 12 13 14 <- index
# 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 <- sequence
#           ^start   ^end
# first window:
# 1 2 3 4 5 6 7 8 9 10 11
#         last window:
#         5 6 7 8 9 10 11 12 13 14 15


chunks = slidingWindow(df2['token'].tolist(),winSize=10)
#now take output from generator and concatenate into a 'document'
window = []
for each in chunks:
    window.append(' '.join(each))
len(window)
len(df2)
window[1]
window[215086]
#repeat the first and final windows (first 5 and last 5 tokens will have off-center windows)
repeats_start = list(np.repeat(window[0], 5))
repeats_end = list(np.repeat(window[215086], 5))
repeats_start.extend(window)
repeats_start[0]
repeats_start[6]
repeats_start[215091]
len(repeats_start)
window=repeats_start
window.extend(repeats_end)
window[0]
window[5]
window[6]
len(window)
window[215090]
window[215091]
window[215096]
df3 = df2
df3['window'] = window

#old version
# make window (from: https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/)
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
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]

chunks = slidingWindow(df2['token'].tolist(),winSize=10)
#now take output from generator and concatenate into a 'document'
window = []
for each in chunks:
    window.append(' '.join(each))
#repeat the final window to give the last few tokens a window (it will be off-center)
repeats = list(np.repeat(window[215087], 9))
window.extend(repeats)
df3 = df2
df3['window'] = window

# make window (from: https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/)
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
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]

chunks = slidingWindow(df2['token'].tolist(),winSize=10)
#now take output from generator and concatenate into a 'document'
window = []
for each in chunks:
    window.append(' '.join(each))
#repeat the final window to give the last few tokens a window (it will be off-center)
repeats = list(np.repeat(window[215087], 9))
window.extend(repeats)
df3 = df2
df3['window'] = window


#troubleshooting



# make window (from: https://scipher.wordpress.com/2010/12/02/simple-sliding-window-iterator-in-python/)
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
    numOfChunks = int(((len(sequence) - winSize) / step) + 1)
    # Do the work
    for i in range(0, numOfChunks * step, step):
        yield sequence[i:i + winSize]

chunks = slidingWindow(df2['token'].tolist(),winSize=10)
print(next(chunks))
#above works!
#now take output from generator and concatenate into a 'document'
# https://www.programiz.com/python-programming/generator
print(' '.join(next(chunks)))
#above also works
for each in chunks:
    print(each)
#above works
for each in chunks:
    print(' '.join(each))
#above also works
window = []
for each in chunks:
    window.append(each)
#above also works
window = []
for each in chunks:
    window.append(' '.join(each))
#still works
window[3] #check it here
len(window)
len(df2)
window[215087]
#windowing stops before the end of df2
df2.iloc[215087:215097]
#repeat the final window 9 times to make lengths equal
repeats = list(np.repeat(window[215087],9))
window2 = list(window)
len(window2)
len(repeats)
window2.extend(repeats)
len(window2)
window2[215097]
#it worked. problem was that I wrote window2 = window2.extend(repeats), which doesn't work
df3 = df2
df3['window'] = window


len(df3)
len(window2)

type(repeats)
type(window)
for i in repeats:
    window2 = window
    window.append(i)
    return
len(window)
window[215089]
print(window2)

language = ['French', 'English']
# another list of language
language1 = ['Spanish', 'Portuguese']
# appending language1 elements to language
language.extend(language1)


for i in list(range(1, 10)):
    window2 = window.append(window[215087])
window2[1]
window2 = window[1:len(df2['token'].tolist())]


df3 = df2
df3.assign(window = window)
df3['window'] = 4

windows = []
for i in [1:len(df2['token'].tolist())]:
    print(next(chunk))
    windows.append(chunk)
    df = windows

df2


a_list = []
b_list = []
for data in my_data:
    a, b = process_data(data)
    a_list.append(a)
    b_list.append(b)
df = pd.DataFrame({'A': a_list, 'B': b_list})
del a_list, b_list


for chunk in chunks:
    a, b
    df3 = data.append(pd.DataFrame({'window':chunk}, index=[0]))

print(df3)


df3 = list(' '.join(chunks))

for chunk in chunks:
    df = pd.concat([chunk])


for chunk in chunks:
    print(df2['token'].tolist())



len(df2['token'].tolist())

for chunk in chunks:
    print(df2['token'].tolist())


from itertools import islice
def window()


from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

# implement tf-idf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

cndf.head

cv=CountVectorizer()
vectorizer = TfidfVectorizer()
analyze = vectorizer.build_analyzer()

# drop word embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')]
# reset index
df2 = df2.reset_index()

docs=["the house had a tiny little mouse",
"the cat saw the mouse",
"the mouse ran away from the house",
"the cat finally ate the mouse",
"the end of the mouse story"
]


df2_test1 =  df2.loc[df2['note'] == df2.at[1, 'note'], :]
df2_test1.head()


df2.loc[1:10, ['token', 'index', 'start', 'end', 'Resp_imp', 'note']]

pd.crosstab(df2['Resp_imp'], "count")




df2.columns

df2[1, 'note']

df2.head

df2.index.duplicated

df2.at[1, 'note']

df2.at[1, 'n]

df2.loc([0], ['note'])
df.loc([0], ['Country'])

data = {'Country': ['Belgium', 'India', 'Brazil'],
        'Capital': ['Brussels', 'New Delhi', 'Brasília'],
        'Population': [11190846, 1303171035, 207847528]}
data = pd.DataFrame(data, columns=['Country', 'Capital', 'Population'])
data[1:]
data.iloc([0],[0])
data.loc([0], ['Country'])


#AL pipeline (uses output from _08_window_classifier.py neural net:
"""
Now figure out the winner and ingest the unlabeled notes
"""

print('starting entropy search')

hpdf = pd.read_json(f"{ALdir}hpdf.json")
winner = hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]

# load it
best_model = pd.read_pickle(f"{ALdir}model_batch4_{int(winner.idx)}.pkl")
model = makemodel(*best_model['hps'])
model.set_weights(best_model['weights'])

# find all the notes to check
notefiles = [i for i in os.listdir(f"{outdir}embedded_notes/")]
# lose the ones that are in the trnotes:
trstubs = ["_".join(i.split("_")[-2:]) for i in trnotes]
testubs = ["_".join(i.split("_")[-2:]) for i in tenotes]
notefiles = [i for i in notefiles if (i not in trstubs) and (i not in testubs) and ("DS_Store" not in i)]
# and lose the ones that aren't 2018
notefiles = [i for i in notefiles if int(i.split("_")[2][1:]) <= 12]
# lose the ones that aren't in the cndf
# the cndf was cut on July 14, 2020 to only include notes from PTs with qualifying ICD codes from the 12 months previous
cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
    cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
cndf_notes = ("embedded_note_m"+cndf.month.astype(str)+"_"+cndf.PAT_ID+".pkl").tolist()
notefiles = list(set(notefiles) & set(cndf_notes))

def h(x):
    """entropy"""
    return -np.sum(x * np.log(x), axis=1)


# now loop through them, normalize, and predict
def get_entropy_stats(i, return_raw=False):
    try:
        start = time.time()
        note = pd.read_pickle(f"{outdir}embedded_notes/{i}")
        note[str_varnames + embedding_colnames] = scaler.transform(note[str_varnames + embedding_colnames])
        note['note'] = "foo"
        if best_model['hps'][-1] is False:  # corresponds with the semipar argument
            Xte = tensormaker(note, ['foo'], str_varnames + embedding_colnames, best_model['hps'][0])
        else:
            Xte_np = tensormaker(note, ['foo'], embedding_colnames, best_model['hps'][0])
            Xte_p = np.vstack([note[str_varnames] for i in ['foo']])

        pred = model.predict([Xte_np, Xte_p] if best_model['hps'][5] is True else Xte)
        hmat = np.stack([h(i) for i in pred])
        end = time.time()

        out = dict(note=i,
                   hmean=np.mean(hmat),
                   # compute average entropy, throwing out lower half
                   hmean_top_half=np.mean(hmat[hmat > np.median(hmat)]),
                   # compute average entropy, throwing out those that are below the (skewed) average
                   hmean_above_average=np.mean(hmat[hmat > np.mean(hmat)]),
                   # maximum
                   hmax=np.max(hmat),
                   # top decile average
                   hdec=np.mean(hmat[hmat > np.quantile(hmat, .9)]),
                   # the raw predictions
                   pred=pred,
                   # time
                   time = end-start
                   )
        return out
    except Exception as e:
        print(e)
        print(i)

if "entropies_of_unlableled_notes.pkl" not in os.listdir(ALdir):
    edicts = []
    N = 0
    for i in notefiles:
        if f"pred{i}.pkl" not in os.listdir(f"{ALdir}ospreds/"):
            r = get_entropy_stats(i)
            write_pickle(r, f"{ALdir}ospreds/pred{i}.pkl")
        else:
            r = read_pickle(f"{ALdir}ospreds/pred{i}.pkl")
        r.pop("pred")
        print(r)
        edicts.append(r)
        print(i)
        N += 1
        print(N)

    # res = pd.concat([res, pd.DataFrame([i for i in edicts if i is not None])])
    res = pd.DataFrame([i for i in edicts if i is not None])
    res.to_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")
else:
    res = pd.read_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")


colnames = res.columns[1:].tolist()
fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
for i in range(5):
    for j in range(5):
        if i == j:
            ax[i, j].hist(res[colnames[i]])
            ax[i, j].set_xlabel(colnames[i])
        elif i > j:
            ax[i, j].scatter(res[colnames[i]], [res[colnames[j]]], s=.5)
            ax[i, j].set_xlabel(colnames[i])
            ax[i, j].set_ylabel(colnames[j])
plt.tight_layout()
fig.savefig(f"{ALdir}entropy_summaries.pdf")
plt.show()

# pull the best notes
cndf['note'] = "embedded_note_m"+cndf.month.astype(str)+"_"+cndf.PAT_ID+".pkl"
res = res.merge(cndf[['note', 'combined_notes']])

best = res.sort_values("hmean_above_average", ascending = False).head(30)
best['PAT_ID'] = best.note.apply(lambda x: x.split("_")[3][:-4])
best['month'] = best.note.apply(lambda x: x.split("_")[2][1:])
cndf['month'] = cndf.LATEST_TIME.dt.month

selected_notes = []
for i in range(len(best)):
    ni = cndf.combined_notes.loc[(cndf.month == int(best.month.iloc[i])) & (cndf.PAT_ID == best.PAT_ID.iloc[i])]
    assert len(ni) == 1
    selected_notes.append(ni)

for i, n in enumerate(selected_notes):
    # assert 5==0, 'SET THIS BACK TO 25 NEXT TIME YOU RUN IT'
    fn = f"AL{batchstring}_v2{'ALTERNATE' if i >24 else ''}_m{best.month.iloc[i]}_{best.PAT_ID.iloc[i]}.txt"
    print(fn)
    write_txt(n.iloc[0], f"{ALdir}{fn}")


# post-analysis
hpdf.to_csv(f"{ALdir}hpdf_for_R.csv")

# get the files
try:
    os.mkdir(f"{ALdir}best_notes_embedded")
except Exception:
    pass
if 'crandrew' in os.getcwd():
    assert 5==0 # if you ver need to make this work again, fix it to not rely on grace
    for note in best.note:
        cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/saved_models/AL{batchstring}/ospreds/pred{note}.pkl" \
              f" {ALdir}ospreds/"
        os.system(cmd)
    for note in best.note:
        cmd = f"scp andrewcd@grace.pmacs.upenn.edu:/media/drv2/andrewcd2/frailty/output/embedded_notes/{note}" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)
elif 'hipaa_garywlab' in os.getcwd():
    for note in best.note:
        cmd = f"cp {ALdir}/ospreds/pred{note}.pkl" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)
    for note in best.note:
        cmd = f"cp /project/hipaa_garywlab/frailty/output/embedded_notes/{note}" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)


predfiles = os.listdir(f"{ALdir}best_notes_embedded")
predfiles = [i for i in predfiles if "predembedded" in i]
enotes = os.listdir(f"{ALdir}best_notes_embedded")
enotes = [i for i in enotes if "predembedded" not in i]

j = 0
for j, k in enumerate(predfiles):
    p = read_pickle(f"{ALdir}best_notes_embedded/{predfiles[j]}")
    ID = re.sub('.pkl', '', "_".join(predfiles[j].split("_")[2:]))
    emat = np.stack([h(x) for x in p['pred']]).T
    emb_note = read_pickle(f"{ALdir}best_notes_embedded/{[x for x in enotes if ID in x][0]}")
    fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
    for i in range(4):
        ax[i].plot(p['pred'][i][:,0], label='neg')
        ax[i].plot(p['pred'][i][:,2], label='pos')
        hx = h(p['pred'][i])
        ax[i].plot(hx+1, label='entropy')
        ax[i].legend()
        ax[i].axhline(1)
        ax[i].set_ylabel(out_varnames[i])
        ax[i].set_ylim(0, 2.1)
        maxH = np.argmax(emat[:,i])
        span = emb_note.token.iloc[(maxH - best_model['hps'][0]//2):(maxH + best_model['hps'][0]//2)]
        string = " ".join(span.tolist())
        ax[i].text(maxH, 2.1, string, horizontalalignment='center')
    plt.tight_layout()
    plt.show()
    fig.savefig(f"{ALdir}best_notes_embedded/predplot_w_best_span_{enotes[j]}.pdf")


