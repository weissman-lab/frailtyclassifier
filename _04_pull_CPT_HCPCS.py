'''
This script will pull CPT and HCPCS codes, as well as ICD9 codes to replicate the Kim et al article's index.
It'll comile a data frame with
    -patient ID
    -code
    -code type
    -date
It'll do so for all patients represented in the concatenated notes data frame

cleaning CPT codes:
--trailing single letters seem to indicate the status of something.  for example, CSN 26198351
    has CPT 85651 and 85651C, and they're both described in ORDER_PROC.DESCRIPTION as SEDIMENTATION
    but they have different status -- one is "sent" and the other is "completed"
    this is also bourne out by they CPT 85610 and 85610A, which is a prothombrin test, requested
    and subsequently completed
--weirdly formatted CPT codes seem to be duplicates of properly-formatted CPT codes.  C3050065
    is described as the same PROTHOMBRIN test as the previous one
    BUT not always.  there are singleton bad-formatted ones.
'''

import pandas as pd
import os
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, \
    make_sql_string, query_filtered_with_temp_tables, nrow
import re
import numpy as np
import time
import sys
import multiprocessing as mp
import copy
import matplotlib.pyplot as plt

# preferences
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

# connect to the database
clar_conn = get_clarity_conn("/Users/crandrew/Documents/clarity_creds_ACD.yaml")

datadir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/data/"
outdir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/"

# load the combined notes:
cndf = pd.read_pickle(f'{outdir}conc_notes_df.pkl')
uids = cndf.PAT_ID.unique().tolist()
len(uids)

if "cpt_hcpcs_df.pkl" not in os.listdir(outdir):

    # query the cpt and HCPCS codes:
    base_query = """
    select
            op.PAT_ENC_CSN_ID as CSN
        ,   ceo.CPT_CODE as CPT
        ,   op.PAT_ID
        ,   ceo.CODE_TYPE_C as TYPE
        , 	op.ORDERING_DATE
        , 	op.PROC_ID
        ,	op.DESCRIPTION
        ,   maxdate = Max(ceo.CONTACT_DATE_REAL)
    from ORDER_PROC as op
    inner join CLARITY_EAP_OT as ceo on ceo.PROC_ID = op.PROC_ID
    """
    fdict = dict(PAT_ID={"vals": [], "foreign_table":"op",
                              "foreign_key":"PAT_ID"},
                 CODE_TYPE_C={"vals": [1,2], "foreign_table":"ceo",
                              "foreign_key":"CODE_TYPE_C"})
    footers = """
    where ceo.CPT_CODE is not null
    and op.ORDERING_DATE > '2017-01-01'
    group by op.PAT_ENC_CSN_ID, op.PAT_ID, ceo.CPT_CODE, ceo.CODE_TYPE_C, op.PROC_ID, op.ORDERING_DATE, op.DESCRIPTION, op.PROC_ID
    """

    def wrapper(ids):
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        q += footers
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        return out

    UIDs = cndf.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i*1000):((i+1)*1000)] for i in range(len(UIDs)//1000+1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    cptlist = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    chdf = pd.concat(cptlist)

    def get_bad_cpt():
        return chdf.CPT[(chdf.TYPE == 1) & ((chdf.CPT.apply(len) != 5) | (chdf.CPT.str.isdigit() == False))]


    def check_nbad():
        badcpt = get_bad_cpt()
        print(f'{len(badcpt) / sum(chdf.TYPE == 1) * 100}% remain bad')


    check_nbad()


    # deal with bad cpt codes
    # first trim off trailing letters
    def trim_trailing_letter(x):
        if (len(x) == 6):
            if x[-1].isalpha():
                if x[:5].isdigit():
                    return x[:5]
        return x


    chdf.CPT = chdf.CPT.apply(trim_trailing_letter)
    check_nbad()
    # now for the bad ones, see if they match the descriptions of any good ones
    badcpt = get_bad_cpt().unique()
    change_dict = {}
    for i in range(len(badcpt)):
        desc = chdf.DESCRIPTION[chdf.CPT == badcpt[i]].unique()
        print(desc)
        if len(desc) == 1:
            desc = desc[0]
            equivs = chdf.CPT[(chdf.DESCRIPTION == desc) & (chdf.CPT != badcpt[i])].unique()
            equivs = [j for j in equivs if ((len(j) == 5) & j.isdigit())]
            print(equivs)
            if len(equivs) == 1:
                change_dict[badcpt[i]] = equivs[0]

    for i in list(change_dict.keys()):
        chdf.CPT[chdf.CPT == i] = change_dict[i]
    check_nbad()

    # now try the same thing by proc ID
    badcpt = get_bad_cpt().unique()
    change_dict = {}
    for i in range(len(badcpt)):
        pid = chdf.PROC_ID[chdf.CPT == badcpt[i]].unique()
        if len(pid) == 1:
            pid = pid[0]
            equivs = chdf.CPT[(chdf.PROC_ID == pid) & (chdf.CPT != badcpt[i])].unique()
            equivs = [j for j in equivs if ((len(j) == 5) & j.isdigit())]
            print(equivs)
            if len(equivs) == 1:
                change_dict[badcpt[i]] = equivs[0]
    for i in list(change_dict.keys()):
        chdf.CPT[chdf.CPT == i] = change_dict[i]
    check_nbad()


    # now the same for HCPCS
    def get_bad_hcpcs():
        return chdf.CPT[(chdf.TYPE == 2) & ((chdf.CPT.apply(len) != 5) |
                                            (chdf.CPT.apply(lambda x: x[0].isalpha()) == False) |
                                            (chdf.CPT.apply(lambda x: x[1:].isdigit()) == False))]


    def check_nbad_hcpcs():
        badhcpcs = get_bad_hcpcs()
        print(f'{len(badhcpcs) / sum(chdf.TYPE == 2) * 100}% remain bad')


    check_nbad_hcpcs()
    '''
    it looks clear that some CPT codes are actually HCPCS codes
    chdf[(chdf.TYPE == 2) & chdf.CPT.isin(get_bad_hcpcs().unique())]
    chdf.CPT[chdf.TYPE == 2].unique()
    '''
    # convert the ones that are CPT formatted to type = 1, checking to make sure that they have the same description as
    # those that are coded as CPT already
    cpt_formatted_hcpcs = [i for i in get_bad_hcpcs() if ((len(i) == 5) & (i.isdigit() == True))]
    for i in cpt_formatted_hcpcs:
        chdf.loc[(chdf.TYPE == 2) & (chdf.CPT == i), "TYPE"] = 1

    badhcpcs = get_bad_hcpcs().unique()
    change_dict = {}
    for i in range(len(badhcpcs)):
        desc = chdf.DESCRIPTION[chdf.CPT == badhcpcs[i]].unique()
        print(desc)
        if len(desc) == 1:
            desc = desc[0]
            equivs = chdf.CPT[(chdf.DESCRIPTION == desc) & (chdf.CPT != badhcpcs[i])].unique()
            equivs = [j for j in equivs if ((len(j) == 5) & j[0].isalpha())]
            print(equivs)
            if len(equivs) == 1:
                change_dict[badhcpcs[i]] = equivs[0]

    for i in list(change_dict.keys()):
        chdf.CPT[chdf.CPT == i] = change_dict[i]
    check_nbad_hcpcs()

    badhcpcs = get_bad_hcpcs().unique()
    change_dict = {}
    for i in range(len(badhcpcs)):
        desc = chdf.PROC_ID[chdf.CPT == badhcpcs[i]].unique()
        print(desc)
        if len(desc) == 1:
            desc = desc[0]
            equivs = chdf.CPT[(chdf.PROC_ID == desc) & (chdf.CPT != badhcpcs[i])].unique()
            equivs = [j for j in equivs if ((len(j) == 5) & j[0].isalpha())]
            print(equivs)
            if len(equivs) == 1:
                change_dict[badhcpcs[i]] = equivs[0]

    # save the output
    chdf.to_pickle(f'{outdir}cpt_hcpcs_df.pkl')
else:
    chdf = pd.read_pickle(f'{outdir}cpt_hcpcs_df.pkl')




'''
Pull the ICD9 codes
'''

if "icd9_df.pkl" in os.listdir(outdir):
    icd9df = pd.read_pickle(f'{outdir}icd9_df.pkl')
else:
    base_query = """
    select
            ped.PAT_ID
        ,   eci.CODE as ICD9
        ,   ped.CONTACT_DATE
    from PAT_ENC_DX as ped
    inner join EDG_CURRENT_ICD9 as eci on ped.DX_ID = eci.DX_ID
    """
    fdict = dict(PAT_ID={"vals": [], "foreign_table": "ped",
                         "foreign_key": "PAT_ID"})
    footers = "where ped.CONTACT_DATE > '2017-01-01'"


    def wrapper(ids):
        fd = copy.deepcopy(fdict)
        fd['PAT_ID']['vals'] = ids
        q = query_filtered_with_temp_tables(base_query, fd, rstring=str(np.random.choice(10000000000)))
        q += footers
        out = get_from_clarity_then_save(q, clar_conn=clar_conn)
        return out


    UIDs = cndf.PAT_ID.unique().tolist()
    UIDs.sort()
    chunks = [UIDs[(i * 1000):((i + 1) * 1000)] for i in range(len(UIDs) // 1000 + 1)]

    pool = mp.Pool(processes=mp.cpu_count())
    start = time.time()
    icdlist = pool.map(wrapper, chunks, chunksize=1)
    print(time.time() - start)
    pool.close()

    icd9df = pd.concat(icdlist)
    icd9df.to_pickle(f'{outdir}icd9_df.pkl')



'''
Combine the two data frames.  First add a consistent type column, then push date to month
Save it.  Filter for desired dates before pivoting, to save space.
'''
chdf.head()
chdf.dtypes
icd9df.head()
icd9df.dtypes
icd9df['TYPE'] = "icd9"
chdf.TYPE = chdf.TYPE.astype(str)
chdf['TYPE'][chdf['TYPE'] == "1"] = "cpt"
chdf['TYPE'][chdf['TYPE'] == "2"] = "hcpcs"
icd9df=icd9df.rename(columns={"ICD9":"CODE", "CONTACT_DATE":"DATE"})
chdf=chdf.rename(columns={"CPT":"CODE", "ORDERING_DATE":"DATE"})

df = pd.concat([chdf[["PAT_ID", "DATE", "CODE", "TYPE"]],
                icd9df[["PAT_ID", "DATE", "CODE", "TYPE"]]])
df = df.loc[df.DATE<"2020-01-01"]

del chdf, icd9df

# convert to month
df.DATE = df.DATE.dt.month + (df.DATE.dt.year*12 - 2017*12)
df.drop_duplicates(inplace=True)
df.to_pickle(f'{outdir}codes_df.pkl')

'''
get unique values of codes/types, then associate with coefs, then merge
'''
ucodes = df[['TYPE', 'CODE']].drop_duplicates()
# lose the ones that aren't interpretable
# have a look at them first...
# ucodes[(ucodes.TYPE == 'cpt') & (ucodes.CODE.apply(len) !=5)]
dropcpt = ucodes.CODE[(ucodes.TYPE == 'cpt') & (ucodes.CODE.apply(len) !=5)]
ucodes = ucodes[~ucodes.CODE.isin(dropcpt)]

drophcpcs = ucodes[(ucodes.TYPE == 'hcpcs') &
       ((ucodes.CODE.apply(len) != 5) |
        (ucodes.CODE.apply(lambda x: x[0].isalpha()) == False) |
        (ucodes.CODE.apply(lambda x: x[1:].isdigit()) == False))]
# noticing here a couple of codes where I can make a post-hoc fix:
df.loc[df.CODE == "J1885A", "CODE"] = "J1885"
df.loc[df.CODE == "J0696A", "CODE"] = "J0696"
ucodes.loc[ucodes.CODE == "J1885A", "CODE"] = "J1885"
ucodes.loc[ucodes.CODE == "J0696A", "CODE"] = "J0696"
ucodes = ucodes[~ucodes.CODE.isin(drophcpcs.CODE)]

# load the kim coefs
kim = pd.read_csv("cpt_hcpcs_kim.csv")
# detect letters in the codes
kim["letter"] = kim.code.apply(lambda x: "".join(re.findall("[a-zA-Z]+", x)))
kim["letter"] = [i[0] if len(i)>0 else "" for i in kim.letter]
# construct range fields
kim["start"] = ""
kim['stop'] = ""
for i in range(nrow(kim.code)):
    if "-" in kim.code[i]:
        kim.start.iloc[i], kim.stop.iloc[i] = kim.code.iloc[i].split("-")
    else:
        kim.start.iloc[i], kim.stop.iloc[i] = kim.code.iloc[i], kim.code.iloc[i]
    kim.start.iloc[i] = re.sub("[a-zA-Z]+", "", kim.start.iloc[i])
    kim.stop.iloc[i] = re.sub("[a-zA-Z]+", "", kim.stop.iloc[i])
# drop the intercept
intercept = kim.coef[kim.name == "Intercept"]
kim = kim[~kim.name.isin(['Intercept'])]
kim.start = kim.start.astype(int)
kim.stop = kim.stop.astype(int)

ucodes.head()
ucodes['coef'] = np.nan
for i in range(nrow(ucodes)):
    targ = ucodes.CODE.iloc[i]
    letter = re.findall("[a-zA-Z]+", targ)
    letter = letter[0] if len(letter)>0 else "" # this only works for single letters, but that's a feature rather than a bug
    number = float(re.sub("[a-zA-Z]+","", targ)) if any(re.findall("[0-9]", targ)) else -999
    if number != -999:
        coef = kim[(kim.letter == letter) &
                    (kim.start <= number) &
                    (kim.stop >= number) &
                    (kim.type == ucodes.TYPE.iloc[i])].coef
        if nrow(coef) > 0:
            assert nrow(coef) == 1
            ucodes.coef.iloc[i] = coef.iloc[0]
            print(coef.iloc[0])

    isinstance(re.findall("[a-zA-Z]+", "5"), str)

ucodes = ucodes[~np.isnan(ucodes.coef)]

df = df.merge(ucodes, how = 'inner')
df.head()

def f(ID): # kim aggregation function
    x = df.loc[df.PAT_ID == ID]
    # months = [i for i in x.DATE.sort_values().unique() if i > 12]
    months = list(range(13,(df.DATE.max()+1)))
    out = []
    for m in months:
        mdf = x.loc[(x.DATE>(m-12)) & (x.DATE <= m), ['CODE', "TYPE","coef"]].drop_duplicates()
        out.append(dict(DATE = m, score = float(mdf.coef.sum()+intercept)))
    out = pd.DataFrame(out)
    out['PAT_ID'] = x.PAT_ID.iloc[0]
    return out

UIDs = df.PAT_ID.unique().tolist()

pool = mp.Pool(processes=mp.cpu_count())
start = time.time()
kimlist = pool.map(f, UIDs, chunksize=1)
print(time.time() - start)
pool.close()

kimdf = pd.concat(kimlist)
kimdf.to_pickle(f"{outdir}kim_score_df.pkl")

# kimdf = pd.read_pickle(f"{outdir}kim_score_df.pkl")
# kimdf.head()
# kimdf.loc[kimdf.PAT_ID == "003930088"]
#
# chdf.loc[chdf.PAT_ID == '003930088'].sort_values('ORDERING_DATE')
# icd9df.loc[icd9df.PAT_ID == '003930088'].sort_values('CONTACT_DATE')
#
#
# plt.hist(kimdf.score)
# plt.xlabel("kim score")
# plt.show()
#
# X = np.vstack([np.ones(nrow(kimdf)), kimdf.DATE]).T
# y = np.array(kimdf.score)
#
# boot = []
# np.linalg.inv(X.T @ X) @ X.T @ y
# for i in range(1000):
#     s = np.random.choice(nrow(X), nrow(X))
#     Xs = X[s,:]
#     ys = y[s]
#     boot.append(np.linalg.inv(Xs.T @ Xs) @ Xs.T @ ys)
#
# boots = np.vstack(boot)
# boots.shape
# plt.hist(boots[:,1])
# plt.show()
#
