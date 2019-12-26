

'''
This script duplicates much of what was in the previous script, but with a few additions:
- joining multi-word expressions
- outputting "windowed" notes.  Algo:
    initilialize empty list of MRNs who won't be eligible for the next 6 months
    for each month
        define eligible notes as people who haven't had an eligible note in the last 6 months but are otherwise eligible
        add those people to the 6-month list, along with the month in which they were added
        take those notes, and concatenate them with the last 6 months of notes
        process the combined note, and output it


'''

import pandas as pd
import matplotlib.pyplot as plt
from flashtext import KeywordProcessor
import pickle
import re
import multiprocessing
import time
import numpy as np

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"
figdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/figures/"

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

'''
the previous script built a data frame of notes metadata that includes all notes from Jan 2018 onwards
it includes co-morbidities, and only encounters with at least two qualifying diagnoses in the past 12 months
load it.  
initialize two empty data frames with patient ID and time columns.  
    - the first is the windower
    - the second is the running list of note files to generate
loop through months.  at each month:
    - drop people from the windower if they were added more than 6 months ago
    - add people to a temporary list if they are not in the windower and have a note that month
    - append the temporary list to the running list, and to the windower
'''
znotes = pd.read_csv(f"{datadir}notes_metadata_2018.csv")
znotes.ENTRY_TIME = pd.to_datetime(znotes.ENTRY_TIME)
# create month since jan 2018 variable
znotes['month'] = znotes.ENTRY_TIME.dt.month + (znotes.ENTRY_TIME.dt.year - min(znotes.ENTRY_TIME.dt.year)) * 12
# create empty dfs
windower = pd.DataFrame(columns=["PAT_ID", "month"])
running_list = pd.DataFrame(columns=["PAT_ID", "month"])

months = [i for i in range(min(znotes['month']), max(znotes['month']) + 1)]

for m in months:
    windower = windower[(m - windower['month']) < 6]
    tmp = znotes[(znotes["month"] == m) & (~znotes['PAT_ID'].isin(windower['PAT_ID']))][
        ["PAT_ID", "month"]].drop_duplicates()
    windower = pd.concat([windower, tmp], axis=0, ignore_index=True)
    running_list = pd.concat([running_list, tmp], axis=0, ignore_index=True)

# plot notes per month
notes_by_month = running_list.month.value_counts().sort_index()
f = plt.figure()
axes = plt.gca()
axes.set_ylim([0, max(notes_by_month) + 100])
plt.plot(months, notes_by_month, "o")
plt.plot(months, notes_by_month)
plt.xlabel("Months since Jan 2018")
plt.ylabel("Number of notes")
# plt.show()
plt.figure(figsize=(8, 8))
f.savefig(f'{figdir}pat_per_month.pdf')

# make a figure that is number of unique patients with at least two encounters over the past 12 months, by month


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

# load notes.  this is the DF with all the data pulled from clarity.  there are some colums here that aren't in znotes
notes = pd.read_json(f'{datadir}combined_notes_df.json.bz2')
notes["month"] = notes.ENTRY_TIME.dt.month + (notes.ENTRY_TIME.dt.year - 2017) * 12  # note here the hard-coded 2017.


def proc(j):
    pi, mi = running_list.PAT_ID[j], running_list.month[
        j] + 12  # the "+12 is there because the running list started in 2018"
    # slice the df
    ni = notes[(notes.PAT_ID == pi) & ((mi - notes.month) < 6) & ((mi - notes.month) >= 0)]
    ni = ni.sort_values(by=["ENTRY_TIME"], ascending=False)
    # process the notes
    comb_notes = [identify_mwes(i, macer) for i in ni.NOTE_TEXT]
    comb_string = ""
    for i in list(range(len(comb_notes))):
        comb_string = comb_string + joiner + str(ni.ENTRY_TIME.iloc[i]) + joiner + comb_notes[i]

    wds = re.split(" |\n", comb_string)
    wds = [i.lower() for i in wds if i != ""]
    comb_note_dict_i = dict(PAT_ID=ni.PAT_ID.iloc[0],
                            LATEST_TIME=ni.ENTRY_TIME.iloc[0],
                            CSNS=",".join(ni.PAT_ENC_CSN_ID.astype(str).to_list()),
                            n_notes=ni.shape[0],
                            n_words=len(wds),
                            u_words=len(set(wds)),
                            combined_notes=comb_string)
    return comb_note_dict_i


pool = multiprocessing.Pool(processes=16)
start = time.time()
dictlist = pool.map(proc, range(running_list.shape[0]), chunksize=1)
print(time.time() - start)
pool.close()

# make a data frame
ds = dictlist
d = {}
for k in dictlist[1].keys():
    d[k] = tuple(d[k] for d in ds)
conc_notes_df = pd.DataFrame(d)
# remove multple newlines
conc_notes_df.combined_notes = conc_notes_df.combined_notes.apply(lambda x: re.sub("\n\n+", "\n\n", xx))
conc_notes_df.to_pickle(f'{outdir}conc_notes_df.pkl')

conc_notes_df['month'] = conc_notes_df.LATEST_TIME.dt.month + (
        conc_notes_df.LATEST_TIME.dt.year - min(conc_notes_df.LATEST_TIME.dt.year)) * 12
months = list(set(conc_notes_df.month))
months.sort()

def plotfun(var, yaxt, q=False):
    f = plt.figure()
    axes = plt.gca()
    if q:
        qvec = [np.quantile(conc_notes_df[var][conc_notes_df.month == i], [.25, .5, .75]).reshape(1, 3) for i in months]
        qmat = np.concatenate(qvec, axis=0)
        axes.set_ylim([0, np.max(qmat)])
        plt.plot(months, qmat[:, 1], "C1", label="median")
        plt.plot(months, qmat[:, 0], "C2", label="first quartile")
        plt.plot(months, qmat[:, 2], "C2", label="third quartile")
    else:
        sdvec = [np.std(conc_notes_df[var][conc_notes_df.month == i]) for i in months]
        muvec = [np.mean(conc_notes_df[var][conc_notes_df.month == i]) for i in months]
        axes.set_ylim([0, max(np.array(muvec) + np.array(sdvec))])
        plt.plot(months, muvec, "C1", label="mean")
        plt.plot(months, np.array(muvec) + np.array(sdvec), "C2", label="+/- 1 sd")
        plt.plot(months, np.array(muvec) - np.array(sdvec), "C2")
    plt.xlabel("Months since Jan 2018")
    plt.ylabel(yaxt)
    axes.legend()
    # plt.show()
    plt.figure(figsize=(8, 8))
    f.savefig(f'{figdir}{var}.pdf')


plotfun("n_notes", "Number of notes per combined note")
plotfun("n_words", "Number of words per combined note", q=True)
plotfun("u_words", "Number of unique words per combined note", q=True)

# numbers of words by number of conc notes
f, ax = plt.subplots()
nnn = list(set(conc_notes_df.n_notes))
pltlist = [conc_notes_df.u_words[conc_notes_df.n_notes == i] for i in nnn if i < 15]
ax.set_title('Unique words by number of concatenated notes')
ax.boxplot(pltlist)
plt.xlabel("Number of notes concatenated together")
plt.ylabel("Number of unique words")
plt.figure(figsize=(8, 8))
f.savefig(f'{figdir}nnotes_by_uwords.pdf')

# pull some random notes
samp = np.random.choice(conc_notes_df.shape[0], 100)
for i in samp:
    fi = f"random_m{conc_notes_df.month.iloc[i]}_{conc_notes_df.PAT_ID.iloc[i]}.txt"
    with open(f'{outdir}/notes_output/random_pull/{fi}', "w") as f:
        f.write(conc_notes_df.combined_notes.iloc[i])






