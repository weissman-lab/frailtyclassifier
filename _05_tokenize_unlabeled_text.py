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

totry = [
    "/share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/",
    "/Users/crandrew/projects/clinical_word_embeddings/",
    "/proj/cwe/built_models/OA_ALL/W2V_300/"
]


for i in totry:
    try:
        embeddings = KeyedVectors.load(f"{i}/w2v_oa_all_300d.bin")
        break
    except:
        pass



# load the concatenated notes
conc_notes_df = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (conc_notes_df.LATEST_TIME.dt.year - 2018) * 12
conc_notes_df = conc_notes_df.loc[conc_notes_df.month <=12]
print(conc_notes_df.shape)
# load the structured data
impdat_dums = pd.read_csv(f"{outdir}impdat_dums.csv")
impdat_dums.drop(columns = "Unnamed: 0", inplace = True)
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
    span_df['PAT_ID'] = conc_notes_df.PAT_ID.iloc[i]
    span_df['month'] = conc_notes_df.month.iloc[i]
    shp = span_df.shape
    span_df = span_df.merge(impdat_dums)
    assert shp[0] == span_df.shape[0]
    return span_df


def wrapper(i):
    try:
        fn = f"embedded_note_m{conc_notes_df.month.iloc[i]}_{conc_notes_df.PAT_ID.iloc[i]}.pkl"
        if fn not in os.listdir(embedded_outdir):
            x = tokenize(i)
            x.to_pickle(f"{embedded_outdir}{fn}")
    except Exception as e:
        print(i)
        print(e)
        return e

import time
start = time.time()
pool = mp.Pool(mp.cpu_count())
errs = pool.map(wrapper, range(conc_notes_df.shape[0]))
# errs = pool.map(wrapper, np.random.choice(conc_notes_df.shape[0], 1000))
pool.close()
print(errs)
print(time.time() - start)



