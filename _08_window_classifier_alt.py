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


