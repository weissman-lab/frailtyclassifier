import pandas as pd
import os
import multiprocessing as mp
import spacy
from _99_project_module import remove_headers, embeddings_catcher
from gensim.models import KeyedVectors
import numpy as np

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load the spacy stuff
nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])

outdir = f"{os.getcwd()}/output/"
embedded_outdir = f"{outdir}embedded_notes/"

if 'crandrew' in os.getcwd():
    embeddings = KeyedVectors.load("/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin", mmap='r')
else:
    embeddings = KeyedVectors.load("/proj/cwe/built_models/OA_CR/FT_300/ft_oa_corp_300d.bin", mmap='r')


# load the concatenated notes
conc_notes_df = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (conc_notes_df.LATEST_TIME.dt.year - 2018) * 12

# load the structured data
strdat = pd.DataFrame(dict(
    month=conc_notes_df.LATEST_TIME.dt.month + (conc_notes_df.LATEST_TIME.dt.year - 2018) * 12,
    PAT_ID=conc_notes_df.PAT_ID,
    n_comorb=conc_notes_df.n_comorb
))

for i in [i for i in os.listdir(outdir) if "_6m" in i]:
    x = pd.read_pickle(f"{outdir}{i}")
    x = x.drop(columns="PAT_ENC_CSN_ID")
    strdat = strdat.merge(x, how='left')
    print(strdat.shape)

elix = pd.read_csv(f"{outdir}elixhauser_scores.csv")
elix.LATEST_TIME = pd.to_datetime(elix.LATEST_TIME)
elix['month'] = elix.LATEST_TIME.dt.month + (elix.LATEST_TIME.dt.year - 2018) * 12
elix = elix.drop(columns=['CSNS', 'LATEST_TIME', 'Unnamed: 0'])
strdat = strdat.merge(elix, how='left')

# add columns for missing values of structured data
for i in strdat.columns:
    if (strdat[[i]].isna().astype(int).sum() > 0)[0]:
        strdat[[i + "_miss"]] = strdat[[i]].isna().astype(int)
        strdat[[i]] = strdat[[i]].fillna(0)

# save the column names so that I can find them later
str_varnames = list(strdat.columns[2:])


## tokenize
def tokenize(i):
    note = conc_notes_df.combined_notes.iloc[i]
    # tokenize
    res = nlp(note)
    span_df = pd.DataFrame([{"token": i.text, 'length': len(i.text_with_ws)} for i in res])
    span_df['end'] = span_df.length.cumsum().astype(int)
    span_df['start'] = span_df.end - span_df.length
    assert int(span_df.end[-1:]) == len(note)
    span_df = remove_headers(span_df)
    # embed
    Elist = [embeddings_catcher(i, embeddings) for i in span_df.token]
    Emat = pd.DataFrame(np.stack(Elist))
    Emat.columns = ['identity_' + str(i) for i in range(300)]
    span_df = pd.concat([span_df.reset_index(drop=True), Emat], axis=1)
    # add the structured data
    str_i = strdat.loc[(strdat.PAT_ID == conc_notes_df.PAT_ID.iloc[i]) & (strdat.month == conc_notes_df.month.iloc[i])]
    str_rep = str_i.iloc[np.full(span_df.shape[0], 0)].reset_index(drop=True)
    span_df = pd.concat([span_df, str_rep], axis=1)
    return span_df


def wrapper(i):
    try:
        x = tokenize(i)
        fn = f"embedded_note_m{conc_notes_df.month.iloc[i]}_{conc_notes_df.PAT_ID.iloc[i]}.pkl"
        x.to_pickle(f"{embedded_outdir}{fn}")
    except Exception as e:
        return e

import time
start = time.time()
pool = mp.Pool(mp.cpu_count())
# errs = pool.map(wrapper, range(conc_notes_df.shape[0]))
errs = pool.map(wrapper, np.random.choice(conc_notes_df.shape[0], 1000))

pool.close()
print(time.time() - start)
