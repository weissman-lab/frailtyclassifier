
# Implementation of sampling strategy for frailty classifier project
## Andrew Crane-Droesch
## 14 July 2020

This doc will walk through the first two scripts in the `frailty` repo, as of 14 July, and explain how relevant pieces of them implement the sampling strategy.  The goal is to surface any non-shared understanding of the eligibility criteria for the study, so that understanding can be harmonized.

I'll quote long sections of code, reverting to plain text where there is something to emphasize or note.  With that:

## `_01_pull_data.py`

This script pulls the data.

```python
'''
This script pulls the data.
It begins by getting a list of all patient IDs who have ever had one of our target diagnoses.
Armed with that list, the next query pulls all of the notes from their qualifying encounters.
Finally, we pull info on transplants, and use that to filter-out encounters that happened post-transplant.
'''

# libraries and imports
import pandas as pd
import os
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, combine_notes_by_type, \
    query_filtered_with_temp_tables, write_txt
import re
import time
import multiprocessing as mp
import copy
import numpy as np

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# connect to the database
clar_conn = get_clarity_conn("/Users/crandrew/Documents/clarity_creds_ACD.yaml")

# set a global data dir for PHI
datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"


# get the diagnoses data frame.  look first in the data directory.
if "diagnosis_df.pkl" in os.listdir(datadir):
    diagnosis_df = pd.read_pickle(f"{datadir}diagnosis_df.pkl")
else:
    diagnosis_df = get_from_clarity_then_save(
        query=open("_8_diagnosis_query.sql").read(),
        clar_conn=clar_conn
    )
    # subset them to the correct set of diagnostic codes
    chronic_regex = '^(J44\.[0,1,9]|J43\.[0,1,2,8,9]|J41\.[0,1,8]|J42|J84\.10|D86|J84|M34\.81|J99\.[0,1])'
    diagnosis_df = diagnosis_df[diagnosis_df['icd10'].str.contains(chronic_regex)]
    diagnosis_df.to_pickle(f"{datadir}diagnosis_df.pkl")
```

Note above the `chronic_regex` -- this defines the ICD's that get pulled.  The diagnosis query itself (see the GH repo) pulls a slightly more expansive set of ICDs, which the `chronic_regex` winnows down.  

Below, I load the notes query (see GH), and then create a dictionary with the specialties, note types, encounter types, and note statuses that we agreed upon.  Note statuses "2" and "3" correspond to "signed" and "addended".

```python

# now get the notes
unique_PIDs = diagnosis_df.PAT_ID.unique().tolist()
unique_PIDs.sort()
base_query = open("_8_patient_notes_query.sql").read()
fdict = dict(PAT_ID={"vals": [], "foreign_table":"pe",
                          "foreign_key":"PAT_ID"},
             SPECIALTY={"vals": ['INTERNAL MEDICINE', 'PULMONARY', 'FAMILY PRACTICE', 'GERONTOLOGY',
                                 'CARDIOLOGY', 'RHEUMATOLOGY', 'Neurology'],
                        "foreign_table": "cd",
                          "foreign_key":"SPECIALTY"},
             NOTE_TYPE={"vals": ['Progress Notes'],
                        "foreign_table": "znti",
                          "foreign_key":"NAME"},
             ENCOUNTER_TYPE={"vals":['Appointment', 'Office Visit', 'Post Hospitalization'],
                             "foreign_table":"zdet",
                          "foreign_key":"NAME"},
             NOTE_STATUS={"vals": ['2', '3'],
                        "foreign_table": "nei",
                          "foreign_key":"note_status_c"}
             )
```

The following does the actual downloading of notes:

```python
# download the notes in small batches, using multiprocessing
# save them to a temp directory, and then concatenate them when done
# to enable this to be restarted, let the function first check and see what's already been done
def get_tbd():
    already_got = [i for i in os.listdir(f"{outdir}/tmp/") if 'notes_' in i]
    gotlist = []
    for i in already_got:
        got = re.sub("notes_|\.pkl", "", i).split("_")
        for j in got:
            gotlist += [j]
    tbd = [i for i in unique_PIDs if i not in gotlist]
    return tbd

tbd = get_tbd()

def wrapper(ids):
    if type(ids).__name__ == 'list':
        assert len(ids) < 6
    if type(ids).__name__ == 'str':
        ids = [ids]
    fd = copy.deepcopy(fdict)
    fd['PAT_ID']['vals'] = ids
    q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
    q += "where pe.ENTRY_TIME >= '2017-01-01'"
    out = get_from_clarity_then_save(q, clar_conn=clar_conn)
    out.to_pickle(f"{outdir}tmp/notes_{'_'.join(ids)}.pkl")



pool = mp.Pool(mp.cpu_count())
start = time.time()
pool.map(wrapper, tbd, chunksize=1)
print(time.time() - start)
pool.close()

# now combine them into a single data  frame, and then split-off a metadata frame,
# it should be smaller and simpler to work with
if "raw_notes_df.pkl" not in os.listdir(datadir):
    tbd = get_tbd()
    assert len(tbd) == 0
    already_got = [i for i in os.listdir(f"{outdir}/tmp/") if 'notes_' in i]
    def proc(path):
        try:
            df = pd.read_pickle(path)
            if df.shape != (0, 0):
                df = combine_notes_by_type(df, CSN = 'CSN', note_type="NOTE_TYPE")
                return df
        except Exception as e:
            message = f"ERROR AT {path}"
            return message

    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    raw_notes_df = pool.map(proc, [outdir + "tmp/" + i for i in already_got], chunksize=1)
    print(time.time() - start)
    pool.close()

    errs = [i for i in raw_notes_df if type(i).__name__ == "str"]
    assert len(errs)==0
    raw_notes_df = pd.concat(raw_notes_df)
    raw_notes_df.to_pickle(f"{datadir}raw_notes_df.pkl")

    # now pull out transplants -- figure out who had a transplant and lose any encounters from before the transplant
    unique_PIDs = raw_notes_df.PAT_ID.unique().tolist()
    unique_PIDs.sort()
    base_query = open("_8_transplant_query.sql").read()
    fdict = dict(PAT_ID={"vals": [], "foreign_table":"ti",
                              "foreign_key":"PAT_ID"}
                 )

    def wrapper(ids):
        if type(ids).__name__ == 'str':
            ids = [ids]
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        q+="\nwhere tc.TX_CLASS_C = 2 and ti.TX_SURG_DT is not NULL"
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        return out

    chunks = [i * 1000 for i in list(range(len(unique_PIDs)//1000+1))]
    chunkids = [unique_PIDs[i:(i+1000)] for i in chunks]
    pool = mp.Pool(1)
    start = time.time()
    txplists = pool.map(wrapper, chunkids, chunksize=1)
    print(time.time() - start)
    pool.close()
    txpdf = pd.concat(txplists)
    txpdf.shape

    raw_notes_df = raw_notes_df.merge(txpdf, how = "left")
    raw_notes_df = raw_notes_df[(raw_notes_df.TX_SURG_DT > raw_notes_df.ENC_DATE) | (raw_notes_df.TX_SURG_DT.isnull())]

    # Keep the ones with at least three visits within the past 12 months
    # so, first sort by MRN and then by date.  then you can proceed linearly.
    raw_notes_df = raw_notes_df.sort_values(["PAT_ID", "ENC_DATE"])

```
Note below:  here is where I implement the criterion that a patient must have at least two qualifying encounters within the past year for them the be "plugged in" to UPHS.  I did it by adding a `time_condition` equal to one where the encounter before last must be within 12 months.  I don't delete those notes.  Rather, I delete unique patients who don't have at least one case where the condition is true.

```python

    # two conditions:
    same_patient_conditon = (raw_notes_df.PAT_ID.shift(periods=2) == raw_notes_df.PAT_ID).astype(int)
    time_conditon = ((raw_notes_df.ENC_DATE - raw_notes_df.ENC_DATE.shift(periods=2)). \
                     dt.total_seconds() / 60 / 60 / 24 / 365 < 1).astype(int)
    # unify them
    condition = same_patient_conditon * time_conditon
    print(raw_notes_df.PAT_ID[condition.astype(bool)].nunique())
    print(raw_notes_df.PAT_ID.nunique())
    raw_notes_df = raw_notes_df[raw_notes_df.PAT_ID.isin(raw_notes_df.PAT_ID[condition.astype(bool)].unique())]

    # pull the dx associated with each CSN
    base_query = open("_8_diagnosis_query_all.sql").read()
    fdict = dict(CSN={"vals": [], "foreign_table": "dx",
                      "foreign_key": "CSN"}
                 )
    def wrapper(ids):
        fd = copy.deepcopy(fdict)
        fd['CSN']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        l = []
        for i in out.CSN.unique().tolist():
            l.append(dict(CSN=i, dxs=','.join(out.icd10[out.CSN == i].unique().tolist())))
        return pd.DataFrame(l)


    csnlist = raw_notes_df.CSN.unique().astype(int).tolist()
    chunks = [i * 1000 for i in list(range(len(csnlist) // 1000 + 1))]
    chunkids = [csnlist[i:(i + 1000)] for i in chunks]
    pool = mp.Pool(mp.cpu_count())
    start = time.time()
    dxdflist = pool.map(wrapper, chunkids, chunksize=1)
    print(time.time() - start)
    pool.close()

    dxdf = pd.concat(dxdflist)
    raw_notes_df = raw_notes_df.merge(dxdf, how='left')
    raw_notes_df.drop(columns="TX_SURG_DT", inplace=True)

    raw_notes_df.to_pickle(f"{outdir}raw_notes_df.pkl")
else:
    raw_notes_df = pd.read_pickle(f"{datadir}raw_notes_df.pkl")


```

## `_preprocessing.py`

This script processes it.

Everything in the next code chunk does cutting of nuisance text:
```python
'''
This script takes the data frame of raw notes generated by the previous script and does the following:
1.  Removes medication lists and other cruft
2.  Joins multiword expressions
3.  "windows" notes:
    initilialize empty list of MRNs who won't be eligible for the next 6 months
    for each month
        define eligible notes as people who haven't had an eligible note in the last 6 months but are otherwise eligible
        add those people to the 6-month list, along with the month in which they were added
        take those notes, and concatenate them with the last 6 months of notes
        process the combined note, and output it
'''

import os
os.chdir("/Users/crandrew/projects/GW_PAIR_frailty_classifier/")
import pandas as pd
import matplotlib.pyplot as plt
from flashtext import KeywordProcessor
import pickle
import re
import multiprocessing as mp
import time
import numpy as np
from _99_project_module import write_txt

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"
figdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/figures/"

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# load the raw notes:
df = pd.read_pickle(f"{outdir}raw_notes_df.pkl")


def identify_spans(**kw):
    '''
    Function to identify sections of text (x) that begin with a start_phrase, end with a stop_phrase,
    contain a cutter_phrase, and don't contain a keeper_phrase
    Finally, it'll run the function extra_function to capture any task-specific corner cases
    '''
    # deal with the arguments
    x = kw.get('x')
    assert x is not None
    start_phrases = kw.get('start_phrases')
    assert start_phrases is not None
    stop_phrases = kw.get('stop_phrases')
    assert stop_phrases is not None
    cutter_phrases = kw.get('cutter_phrases')
    cutter_phrases = cutter_phrases if cutter_phrases is not None else ["SHIVER ME TIMBERS, HERE BE A DEFAULT ARGUMENT"]
    keeper_phrases = kw.get('keeper_phrases')
    keeper_phrases = keeper_phrases if keeper_phrases is not None else ["SHIVER ME TIMBERS, HERE BE A DEFAULT ARGUMENT"]
    extra_function = kw.get('extra_function')
    # begin processing:
    x = x.lower()
    # convert all non-breaking space to regular space
    x = re.sub("\xa0", " ", x)
    # put x back into the kw dict so that the extra functions can use them
    kw['x'] = x
    st_idx = [m.start() for m in re.finditer("|".join(start_phrases), x)]
    stop_idx = [m.start() for m in re.finditer("|".join(stop_phrases), x)]
    spans = []
    try:
        for j in range(len(st_idx)):
            startpos = st_idx[j]
            endpos = next((x for x in stop_idx if x > startpos), None)
            if endpos is not None:
                if j < (len(st_idx)-1): # if j is not the last element being iterated over:
                    if endpos < st_idx[j+1]: # if the end position is before the next start position
                        # implicity, if the above condition doesn't hold, nothing will get appended into spans
                        # and it'll go to the next j
                        span = x[startpos:endpos]
                        # things that it must have to get cut
                        if any(re.finditer("|".join(cutter_phrases), span)):
                            # thing that if it has it can't be cut
                            if not any(re.finditer("|".join(keeper_phrases), span)):
                                if extra_function is not None:
                                    startpos, endpos = extra_function(startpos, endpos, **kw)
                                spans.append((startpos, endpos))
                elif j == (len(st_idx)-1): # if it's the last start index, there better be a following stop index
                    span = x[startpos:endpos]
                    # things that it must have to get cut
                    if any(re.finditer("|".join(cutter_phrases), span)):
                        # thing that if it has it can't be cut
                        if not any(re.finditer("|".join(keeper_phrases), span)):
                            if extra_function is not None:
                                startpos, endpos = extra_function(startpos, endpos, **kw)
                            spans.append((startpos, endpos))
        return spans
    except Exception as e:
        return e


'''
Adult care management risk score
'''
def acm_efun(startpos, endpos, **kw):
    x = kw.get("x")
    stop_phrases = kw.get("stop_phrases")
    '''goal here is to remove text like "Current as_of just now"'''
    # update the endpos to reflect the end, rather than the start, of the end string
    endpos += next(re.finditer("|".join(stop_phrases), x[endpos:len(x)])).end()
    # find first newline after endpos+5.  the 5 is a random number to get past the first \n
    slashnpos = next(re.finditer("\n", x[(endpos+5):len(x)])).end()
    tailstring = x[endpos:(slashnpos-5+endpos+100)]
    # make sure that it contains "current as of", and that it isn't too long
    if ("current as of" in tailstring) and (len(tailstring)<40):
        return startpos, slashnpos-5+endpos
    else:
        return startpos, endpos

'''
CATv2 -- goal here is to end at the tail, rather than the front.  There are different formattings of the final score:
" Score 17 medium impact"
" Score < 10 low impact 10-20 medium impact 21-30 high impact >20 v high impact"
'''
def catv2_efun(startpos, endpos, **kw):
    x = kw.get("x")
    stop_phrases = kw.get("stop_phrases")
    # update the endpos to reflect the end, rather than the start, of the end string
    endpos += next(re.finditer("|".join(stop_phrases), x[endpos:len(x)])).end()
    # find next mention of the word "impact"
    try: # the word "impact" is usually there, but not always
        impactpos = next(re.finditer('impact', x[endpos:len(x)])).end()
        tailstring = x[endpos:(impactpos+endpos)]
        # make sure that it contains "current as of", and that it isn't too long
        if ("score" in tailstring) and (len(tailstring)<100):
            return startpos, impactpos+endpos
        else:
            return startpos, endpos
    except Exception:
        return startpos, endpos


'''
Geriatric wellness
Scanning notes, some of the questionnaires are drimmed
'''
def gw_efun(startpos, endpos, **kw):
    x = kw.get("x")
    stop_phrases = kw.get("stop_phrases")
    # update the endpos to reflect the end, rather than the start, of the end string
    endpos += next(re.finditer("|".join(stop_phrases), x[endpos:len(x)])).end()
    # find next mention of the word 'yes' or 'no' after suicidal ideation
    ynpos = next(re.finditer('yes|no', x[endpos:len(x)])).end()
    tailstring = x[endpos:(ynpos+endpos)]
    # make sure that it contains "current as of", and that it isn't too long
    if ("score" in tailstring) and (len(tailstring)<20):
        return startpos, ynpos+endpos
    else:
        return startpos, endpos


'''
Wrapper function to compute all of the spans
For each note, it'll make a list of dictionaries, with one dictionary for each of the types of questionnaires to be 
removed.
It'll have the form {questionnaire type:argsdict}.
It'll get fed to the identify_spans function, and the output will be a list of spans, in a dictionary keyed by
the questionnaire type
'''
def spans_wrapper(i):
    try:
        '''The argument is the index of the raw notes data frame'''
        # Adult care management risk score
        meta_dict = dict(acm = dict(x=df.NOTE_TEXT.iloc[i],
                                    start_phrases = ["adult care management risk score:"],
                                    stop_phrases = ["patients with an effective medicaid coverage get 1 point."],
                                    cutter_phrases = ["patients with diabetes get 1 point."],
                                    keeper_phrases = None,
                                    extra_function = acm_efun),
                         catv2=dict(x=df.NOTE_TEXT.iloc[i],
                                  start_phrases=["i never cough"],
                                  stop_phrases=["i have no energy at all"],
                                  cutter_phrases=["i do not sleep soundly due to my lung condition"],
                                  keeper_phrases=None,
                                  extra_function=catv2_efun),
                         gw=dict(x=df.NOTE_TEXT.iloc[i],
                                    start_phrases=["geriatrics wellness"],
                                    stop_phrases=["suicidal ideation"],
                                    cutter_phrases=["name a pencil and a watch"],
                                    keeper_phrases=None,
                                    extra_function=gw_efun),
                         meds = dict(x=df.NOTE_TEXT.iloc[i],
                                    start_phrases = ["\nmedication\n", "\nmedications\n", "\nprescriptions\n", "medication:",
                                                     "medications:", "prescriptions:",
                                                     "prescriptions on file"],
                                    stop_phrases = ["allergies\n", "allergies:", "allergy list\n", "allergy list:",
                                                    "-------", "\n\n\n\n",
                                                    "active medical_problems", "active medical problems", "patient active",
                                                    "past surgical history", "past_surgical_history",
                                                    "past medical history", "past_medical_history",
                                                    "review of symptoms", "review_of_symptoms",
                                                    "review of systems", "review_of_systems",
                                                    "family history", "family_history",
                                                    "social history", "social_history", "social hx:",
                                                    "physical exam:", "physical_examination", "physical examination",
                                                    "history of present illness",
                                                    "vital signs",
                                                    "\ni saw ", " i saw ",
                                                    "\npe:",
                                                    "current issues",
                                                    "history\n", "history:"],
                                    cutter_phrases = ["by_mouth", "by mouth"],
                                    keeper_phrases = ['assessment'])
        )
        keylist = list(meta_dict.keys())
        outdict = {}
        for k in keylist:
            outdict[k] = identify_spans(**meta_dict[k])
        return outdict
    except Exception as e:
        return e

# get the spans
pool = mp.Pool(mp.cpu_count())
spanslist = pool.map(spans_wrapper, range(df.shape[0]))
pool.close()

# check for errors
errs = [i for i in range(df.shape[0]) if isinstance(spanslist[i], Exception)]
assert len(errs) == 0


# do the cutting
def highlight_stuff_to_cut(i, do_cutting = False):
    # take the spans and turn them into a dataframe
    dfl = [] #data frame list
    for k in list(spanslist[i].keys()):
        if len(spanslist[i][k])>0:
            for j in range(len(spanslist[i][k])):
                out = dict(variety = k,
                           start = spanslist[i][k][j][0],
                           end = spanslist[i][k][j][1])
                dfl.append(out)
    x = df.NOTE_TEXT.iloc[i]
    if len(dfl)>0:
        sdf = pd.DataFrame(dfl)
        # check and make sure none of the spans overlap
        for j in range(sdf.shape[0]):
            c1 = any((sdf.start.iloc[j] > sdf.start) & (sdf.end.iloc[j] < sdf.end))
            if c1:
                print('overlapping spans!')
                print(i)
                # raise Exception
        # now hilight the note sections
        sdf = sdf.sort_values('start', ascending = False)
        for j in range(sdf.shape[0]):
            left = x[:sdf.start.iloc[j]]
            middle = x[sdf.start.iloc[j]:sdf.end.iloc[j]]
            right = x[sdf.end.iloc[j]:]
            if do_cutting is False:
                x = left +f"\n\n********** BEGIN Cutting {sdf.variety.iloc[j]} *************** \n\n" + \
                    middle + f"\n\n********** END Cutting {sdf.variety.iloc[j]} *************** \n\n" + right
            else:
                x = left + f"---{sdf.variety.iloc[j]}_was_here_but_got_cut----" + right
    return x

'''
There is one overlapping span at 98839.  It's a fairly intractible case.
'''

# get the spans
pool = mp.Pool(mp.cpu_count())
cut_notes = pool.map(highlight_stuff_to_cut, range(df.shape[0]))
pool.close()

checdf = df[['PAT_ID', 'CSN', 'NOTE_ID']]
checdf['notes'] = cut_notes
checdf['acm'] = [len(spanslist[i]['acm'])>0 for i in range(df.shape[0])]
checdf['catv2'] = [len(spanslist[i]['catv2'])>0 for i in range(df.shape[0])]
checdf['gw'] = [len(spanslist[i]['gw'])>0 for i in range(df.shape[0])]
checdf['meds'] = [len(spanslist[i]['meds'])>0 for i in range(df.shape[0])]


x = checdf.loc[checdf.acm == True].iloc[np.random.choice(checdf.acm.sum(), 10)]
for i in range(10):
    write_txt(x.notes.iloc[i], f"{outdir}structured_text_test_output/cutter_tester{x.NOTE_ID.iloc[i]}.txt")

x = checdf.loc[checdf.gw == True].iloc[np.random.choice(checdf.gw.sum(), 10)]
for i in range(10):
    write_txt(x.notes.iloc[i], f"{outdir}structured_text_test_output/cutter_tester{x.NOTE_ID.iloc[i]}.txt")

x = checdf.loc[checdf.catv2 == True].iloc[np.random.choice(checdf.catv2.sum(), 10)]
for i in range(10):
    write_txt(x.notes.iloc[i], f"{outdir}structured_text_test_output/cutter_tester{x.NOTE_ID.iloc[i]}.txt")


# now do the actual cutting
pool = mp.Pool(mp.cpu_count())
cut_notes = pool.starmap(highlight_stuff_to_cut, ((i, True) for i in range(df.shape[0])))
pool.close()

df['NOTE_TEXT'] = cut_notes
```

Here is where I implement the windowing:

```python
'''
initialize two empty data frames with patient ID and time columns.  
    - the first is the windower
    - the second is the running list of note files to generate
loop through months.  at each month:
    - drop people from the windower if they were added more than 6 months ago
    - add people to a temporary list if they are not in the windower and have a note that month
    - append the temporary list to the running list, and to the windower

'''
# create month since jan 2018 variable
df = df[df.ENC_DATE.dt.year>=2017]
df['month'] = df.ENC_DATE.dt.month + (df.ENC_DATE.dt.year - min(df.ENC_DATE.dt.year))*12
# create empty dfs
windower = pd.DataFrame(columns=["PAT_ID", "month"])
running_list = pd.DataFrame(columns=["PAT_ID", "month"])

months = [i for i in range(min(df['month']), max(df['month']) + 1)]

for m in months[12:]:
    windower = windower[(m - windower['month']) < 6]
    tmp = df[(df["month"] == m) & (~df['PAT_ID'].isin(windower['PAT_ID']))][
        ["PAT_ID", "month"]].drop_duplicates()
    windower = pd.concat([windower, tmp], axis=0, ignore_index=True)
    running_list = pd.concat([running_list, tmp], axis=0, ignore_index=True)

# plot notes per month
notes_by_month = running_list.month.value_counts().sort_index().reset_index(drop = True)
f = plt.figure()
axes = plt.gca()
axes.set_ylim([0, max(notes_by_month) + 100])
plt.plot(notes_by_month.index.values, notes_by_month, "o")
plt.plot(notes_by_month.index.values, notes_by_month)
plt.xlabel("Months since Jan 2018")
plt.ylabel("Number of notes")
# plt.show()
plt.figure(figsize=(8, 8))
f.savefig(f'{figdir}pat_per_month.pdf')


'''
armed with the running list of patient IDs, go through the note text, month by month, and concatenate all notes from 
that patient.  join MWEs while at it.
'''
mwe_dict = pickle.load(open("/Users/crandrew/projects/pwe/output/mwe_dict.pkl", 'rb'))
macer = KeywordProcessor()
macer.add_keywords_from_dict(mwe_dict)


def identify_mwes(s, macer):
    return macer.replace_keywords(s)


joiner = "\n--------------------------------------------------------------\n"


def proc(j):
    try:
        pi, mi = running_list.PAT_ID[j], running_list.month[j]  # the "+12 is there because the running list started in 2018"
        # slice the df
        ni = df[(df.PAT_ID == pi) &
                ((mi - df.month) < 6) &
                ((mi - df.month) >= 0)]
        ni = ni.sort_values(by=["ENC_DATE"], ascending=False)
        # process the notes
        comb_notes = [identify_mwes(i, macer) for i in ni.NOTE_TEXT]
        comb_string = ""
        for i in list(range(len(comb_notes))):
            comb_string = comb_string + joiner + str(ni.ENC_DATE.iloc[i]) + \
                          joiner + comb_notes[i]
        # lose multiple newlines
        comb_string = re.sub("\n+", "\n", comb_string)
        # count words
        wds = re.split(" |\n", comb_string)
        wds = [i.lower() for i in wds if i != ""]
        # join the diagnoses
        dxs = list(set((','.join(ni.dxs[~ni.dxs.isnull()].tolist())).split(",")))
        dxs.sort()
        comb_note_dict_i = dict(PAT_ID=ni.PAT_ID.iloc[0],
                                LATEST_TIME=ni.ENC_DATE.iloc[0],
                                CSNS=",".join(ni.CSN.astype(str).to_list()),
                                dxs = dxs,
                                n_comorb = len(dxs),
                                n_notes=ni.shape[0],
                                n_words=len(wds),
                                u_words=len(set(wds)),
                                combined_notes=comb_string)
        return comb_note_dict_i
    except Exception as e:
        return dict(which = j, error = e)


pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
dictlist = pool.map(proc, range(running_list.shape[0]), chunksize=1)
print(time.time() - start)
pool.close()

errs = [i for i in dictlist if "error" in i.keys()]
assert len(errs) == 0

ds = dictlist
d = {}
for k in dictlist[1].keys(): # dict 1 is arbitrary -- it's just pulling the keys
    d[k] = tuple(d[k] for d in ds)
conc_notes_df = pd.DataFrame(d)

# looks for words in the text
low_prob_words = ['gym', 'exercise', 'breathing', 'appetite', 'eating', 'getting around', 'functional status',
                  'PO intake', 'getting around', 'walking', 'running', 'independent']
high_prob_words = ['PO intake', 'weight loss', 'appetite', 'frail', 'frailty', 'weakness', 'feels weak', 'unsteady',
                   'recent fall', 'getting around', 'severe dyspnea', 'functional impairment', 'difficulty walking',
                   'difficulty breathing', 'getting in the way', 'exercise', 'Breathless', 'short of breath',
                   'wheezing', 'delirium', 'dementia', 'incontinence', 'do not resuscitate', 'walker', 'wheelchair',
                   'malnutrition', 'boost']
low_prob_words = [identify_mwes(i, macer) for i in low_prob_words]
high_prob_words = [identify_mwes(i, macer) for i in high_prob_words]

low_prob_regex = '|'.join(low_prob_words)
high_prob_regex = '|'.join(high_prob_words)
# append the hp and lp columns
lp = conc_notes_df.combined_notes.str.contains(low_prob_regex)
hp = conc_notes_df.combined_notes.str.contains(high_prob_regex)
conc_notes_df['highprob'] = hp.values
conc_notes_df['lowprob'] = lp.values
```

And finally, here's what I added today to window out the ICD codes.


```python
'''
July 14 2020:
    remove all concatenated notes that do not have one of the qualifying dx in 
    the 12 months before the latest date in the conc_notes_df
'''

diagnosis_df = pd.read_pickle(f"{datadir}diagnosis_df.pkl")
tm = conc_notes_df[['PAT_ID', 'LATEST_TIME']]
yy= tm.merge(diagnosis_df, how = 'left')
yy.head()
yy['diff'] = yy['LATEST_TIME'] - yy['dx_date']
yy['year'] = (yy['diff'].dt.days > 0) & (yy['diff'].dt.days < 365)
yy = yy.loc[yy.year == True]
yy = yy[["PAT_ID", "LATEST_TIME"]].drop_duplicates()
conc_notes_df = yy.merge(conc_notes_df)

# save
conc_notes_df.to_pickle(f'{outdir}conc_notes_df.pkl')
```
