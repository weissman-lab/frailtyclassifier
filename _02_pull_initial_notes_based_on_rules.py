'''
This is a scratch-ish script to pull some initial notes based on rules.
It'll look in the `znotes` and the `notes text` data frames to pull:
- 50 high-prob notes
- 50 low-prob notes
- 50 random notes
Feedback from review of those notes could lead us to go back and change the queries,
before building other stuff later.
'''

import pandas as pd
import multiprocessing
import os
import json
import time
import re
import numpy as np
import matplotlib.pyplot as plt

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"
figdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/figures/"

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load "znotes" -- the notes metadata
znotes = pd.read_json(f"{datadir}znotes.json.bz2")
znotes = znotes[znotes['ENTRY_TIME'].dt.year >= 2018]

# next load all of the diagnoses, so that we can merge them onto znotes
dx_df = pd.read_json(f'{datadir}all_dx_df.json.bz2')
dx_df.drop_duplicates(inplace=True)

# make sure no commas in the dx df -- this would imply multiple dx per line
has_commas = dx_df[dx_df['CODE'].str.contains(",")]
print(str(has_commas.shape[0]) + " diagnoses have commas")

# we need to associate each note with the diagnoses from the past 12 month.
# I'll do this by first grouping the diagnosis df by patient ID
# then, for each note, get the relevant date, and look up all of the diagnoses within the relevant range of that date
# this should be doable in parallel
dx_groups = dx_df.groupby(by=dx_df.PAT_ID)
UID_dx = list(set(list(dx_groups.groups.keys())))

# write a function that takes a data frame and a date, and returns last-12-month morbidities
# it takes a dict with elements "date" and "df".
# date is the end of the 12-month window, and "df" are all the diagnoses for that particular patient
# it returns nothing if that patient didn't have TWO of the relevant lung diseases in that time
# edit Dec 4:  at least two lung morbidities must be present
chronic_regex = '^(J44\.[0,1,9]|J43\.[0,1,2,8,9]|J41\.[0,1,8]|J42|J84\.10|D86|J84|M34\.81|J99\.[0,1])'


def get_dx_within_window(i, window=1):
    inp = {"df": dx_groups.get_group(znotes.PAT_ID[znotes.PAT_ENC_CSN_ID == i].iloc[0]).drop_duplicates(),
           "date": znotes.ENTRY_TIME[znotes.PAT_ENC_CSN_ID == i]}
    timediff = (inp["date"].iloc[0] - inp["df"].ENTRY_TIME).dt.total_seconds() / 60 / 60 / 24 / 365
    dx_one_year = inp["df"][(timediff > 0) & (timediff < window)].CODE
    if np.sum(dx_one_year.str.contains(chronic_regex)) > 1:
        outdict = {"PAT_ENC_CSN_ID": i,
                   "DX": ','.join(dx_one_year.sort_values().unique())}
        with open(f'{outdir}dx_dicts/_i{i}data.json', 'w') as fp:
            json.dump(outdict, fp)


# make a list of CSNs to feed to the get_dx_within_window function
lgen = znotes.PAT_ENC_CSN_ID[znotes.PAT_ID.isin(UID_dx)].tolist()
donefiles = pd.Series(os.listdir(f'{outdir}dx_dicts/')).replace("_i", "", regex=True) \
    .replace("data.json", "", regex=True)
lser = pd.Series(lgen)
tbd = lser[~lser.isin(donefiles)].tolist()

# run the function.  It takes about 6 Gigs per core, so not using very many processes on my laptop
pool = multiprocessing.Pool(processes=4)
start = time.time()
pool.map(get_dx_within_window, tbd, chunksize=len(tbd) // 4)
print(time.time() - start)
pool.close()

# load the json files and compute the number of dx as the number of commas, plus one, in each dict
TBD = os.listdir(f'{outdir}dx_dicts/')
nTBD = len(TBD)
csnlist = [None] * nTBD
dxcount = [None] * nTBD
dxvec = [None] * nTBD
start = time.time()
for i in range(nTBD):
    with open(f'{outdir}dx_dicts/{TBD[i]}') as x:
        dx = json.loads(x.read())
    csnlist[i] = dx['PAT_ENC_CSN_ID']
    dxcount[i] = len(re.findall(',', dx['DX']))
    dxvec[i] = dx['DX']
print(time.time() - start)

# merge them together
tomerge = pd.DataFrame({'PAT_ENC_CSN_ID': csnlist,
                        'dxcount': dxcount,
                        'dxvec': dxvec})
# plt.hist(np.log10(tomerge.dxcount[tomerge.dxcount >0]))
# plt.show()
znotes = znotes.merge(tomerge, how="inner", on="PAT_ENC_CSN_ID")
znotes.head()

# now go through the notes according to GW's rules for the initial purposive sample
low_prob_words = ['gym', 'exercise', 'breathing', 'appetite', 'eating', 'getting around', 'functional status',
                  'PO intake', 'getting around', 'walking', 'running', 'independent']
high_prob_words = ['PO intake', 'weight loss', 'appetite', 'frail', 'frailty', 'weakness', 'feels weak', 'unsteady',
                   'recent fall', 'getting around', 'severe dyspnea', 'functional impairment', 'difficulty walking',
                   'difficulty breathing', 'getting in the way', 'exercise', 'Breathless', 'short of breath',
                   'wheezing', 'delirium', 'dementia', 'incontinence', 'do not resuscitate', 'walker', 'wheelchair',
                   'malnutrition', 'boost']

low_prob_regex = '|'.join(low_prob_words)
high_prob_regex = '|'.join(high_prob_words)

notes = pd.read_csv(f'{datadir}note_text.csv', index_col=False)
# confirm that the csn's match with those of znotes
notes = notes.loc[notes.PAT_ENC_CSN_ID.isin(znotes.PAT_ENC_CSN_ID)]
print(np.where(notes.PAT_ENC_CSN_ID.values != znotes.PAT_ENC_CSN_ID.values)[0].shape)

# append the hp and lp columns
lp = notes.NOTE_TEXT.str.contains(low_prob_regex)
hp = notes.NOTE_TEXT.str.contains(high_prob_regex)
znotes['highprob'] = hp.values
znotes['lowprob'] = lp.values

# save metadata
znotes.to_csv(f'{datadir}notes_metadata_2018.csv')

# compute number of unique patients per month with notes
ms18 = znotes.ENTRY_TIME.dt.year * 12 - 2018 * 12 + znotes.ENTRY_TIME.dt.month
npm = []
for i in ms18.unique():
    mz = znotes.loc[ms18 == i]
    npm.append(mz.PAT_ID.nunique())

plt.clf()
f = plt.figure()
plt.plot(ms18.unique(), npm)
axes = plt.gca()
axes.set_ylim([0, max(npm)])
plt.xlabel("Months since Jan 2018")
plt.ylabel("Number of unique patients with 2x lung dx in 12 months")
plt.figure(figsize=(8, 8))
f.savefig(f'{figdir}pat_per_month.pdf')
plt.show()

# randomly select 50 from each
lowprob_notes = znotes.PAT_ENC_CSN_ID[(znotes.dxcount <= 5) & (znotes.lowprob == True) & (znotes.highprob == False) \
                                      & (znotes.ENTRY_TIME.dt.year == 2018)]
highprob_notes = znotes.PAT_ENC_CSN_ID[(znotes.dxcount >= 15) & (znotes.lowprob == False) & (znotes.highprob == True) \
                                       & (znotes.ENTRY_TIME.dt.year == 2018)]
other_notes = znotes.PAT_ENC_CSN_ID[(znotes.lowprob == False) & (znotes.highprob == False) \
                                    & (znotes.ENTRY_TIME.dt.year == 2018)]

np.random.seed(8675309)
lp_samp = lowprob_notes.sample(50)
hp_samp = highprob_notes.sample(50)
other_samp = other_notes.sample(50)

for i in range(50):
    # name to save it to
    fn = f'note_lp_{lp_samp.iloc[i]}.txt'
    # the file
    f = open(f'{datadir}notes_output/{fn}', "w")
    # the combined text to put into the file
    metadata = znotes[znotes.PAT_ENC_CSN_ID == lp_samp.iloc[i]]
    to_write = f'<ANNOTATION_METADATA>{str(metadata.to_dict(orient="records"))}</ANNOTATION_METADATA>\
        \n\n{notes.NOTE_TEXT[notes.PAT_ENC_CSN_ID == lp_samp.iloc[i]].tolist()[0]}'
    f.write(to_write)
    f.close()

    # name to save it to
    fn = f'note_hp_{hp_samp.iloc[i]}.txt'
    # the file
    f = open(f'{datadir}notes_output/{fn}', "w")
    # the combined text to put into the file
    metadata = znotes[znotes.PAT_ENC_CSN_ID == hp_samp.iloc[i]]
    to_write = f'<ANNOTATION_METADATA>{str(metadata.to_dict(orient="records"))}</ANNOTATION_METADATA>\
        \n\n{notes.NOTE_TEXT[notes.PAT_ENC_CSN_ID == hp_samp.iloc[i]].tolist()[0]}'
    f.write(to_write)
    f.close()

    # name to save it to
    fn = f'note_other_{other_samp.iloc[i]}.txt'
    # the file
    f = open(f'{datadir}notes_output/{fn}', "w")
    # the combined text to put into the file
    metadata = znotes[znotes.PAT_ENC_CSN_ID == other_samp.iloc[i]]
    to_write = f'<ANNOTATION_METADATA>{str(metadata.to_dict(orient="records"))}</ANNOTATION_METADATA>\
        \n\n{notes.NOTE_TEXT[notes.PAT_ENC_CSN_ID == other_samp.iloc[i]].tolist()[0]}'
    f.write(to_write)
    f.close()

# adding december 11, 2019:
# taking initial notes and joining MWEs

from flashtext import KeywordProcessor
import pickle

mwe_dict = pickle.load(open("/Users/crandrew/projects/pwe/output/mwe_dict.pkl", 'rb'))
macer = KeywordProcessor()
macer.add_keywords_from_dict(mwe_dict)


def identify_mwes(s, macer):
    return macer.replace_keywords(s)

tbd = os.listdir(f'{outdir}notes_output')
for i in list(range(len(tbd))):
    # load the file
    txt = open(f'{outdir}notes_output/{tbd[i]}').read()
    # strip the annotation metadata
    txt_stripped = txt.split("</ANNOTATION_METADATA>")[1]
    # join the mwes
    txt_mwes_joined = identify_mwes(txt_stripped, macer)
    # write them
    fn = re.sub("note_", "note_initial_v2_", tbd[i])
    f = open(f'{outdir}notes_output/{fn}', "w")
    f.write(txt_mwes_joined)
    f.close()
