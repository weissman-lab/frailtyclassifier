'''
This script will pull CPT and HCPCS codes, as well as ICD9 codes to replicate the Kim et al article's index.
It'll comile a data frame with
    -patient ID
    -code
    -code type
    -date
It'll do so for all patients represented in the concatenated notes data frame
'''

import pandas as pd
import os
from _99_project_module import get_clarity_conn, get_from_clarity_then_save, make_sql_string
import re
import numpy as np
import time
import sys

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

# query the cpt and HCPCS codes:
base_query = open("_8_CPT_HCPCS_query.sql").read()
qi = re.sub(":ids", make_sql_string(uids), base_query)



'''
cleaning CPT codes:  
--trailing single letters seem to indicate the status of something.  for example, CSN 26198351
    has CPT 85651 and 85651C, and they're both described in ORDER_PROC.DESCRIPTION as SEDIMENTATION
    but they have different status -- one is "sent" and the other is "completed"
    this is also bourne out by they CPT 85610 and 85610A, which is a prothombrin test, requested
    and subsequently completed
--weirdly formatted CPT codes seem to be duplicates of properly-formatted CPT codes.  C3050065 
    is described as the same PROTHOMBRIN test as the previous one
    BUT not always.  there are singleton bad-formatted ones.  
@@
THe cleaning done below will get skipped if the df is found along the specified path
@@
'''

if "cpt_hcpcs_df.pkl" in os.listdir(outdir):
    chdf = pd.read_pickle(f'{outdir}cpt_hcpcs_df.pkl')
else:
    chdf = get_from_clarity_then_save(qi, clar_conn=clar_conn)


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


'''
Pull the ICD9 codes
'''
base_query = open("_8_ICD9_query.sql").read()
qi = re.sub(":ids", make_sql_string(uids), base_query)

if "icd9_df.pkl" in os.listdir(outdir):
    icd9df = pd.read_pickle(f'{outdir}icd9_df.pkl')
else:
    icd9df = get_from_clarity_then_save(qi, clar_conn=clar_conn)
    icd9df.to_pickle(f'{outdir}icd9_df.pkl')

'''
Combine the two data frames.  First add a consistent type column, then push date to month
Save it.  Filter for desired dates before pivoting, to save space.
'''
chdf.head()
icd9df.head()
icd9df.dtypes
icd9df['TYPE'] = "icd9"
chdf.TYPE = chdf.TYPE.astype(str)
chdf['TYPE'][chdf['TYPE'] == 1] = "cpt"
chdf['TYPE'][chdf['TYPE'] == 2] = "hcpcs"
icd9df['CODE'] = icd9df['ICD9']
chdf['CODE'] = chdf['CPT']
chdf['DATE'] = chdf['ORDERING_DATE']
icd9df['DATE'] = icd9df['CONTACT_DATE']
df = pd.concat([chdf[["PAT_ID", "DATE", "CODE", "TYPE"]],
                icd9df[["PAT_ID", "DATE", "CODE", "TYPE"]]])
df.shape
del chdf, icd9df
# convert to month
df.DATE = df.DATE.dt.month + (df.DATE.dt.year*12 - 2017*12)
df.drop_duplicates(inplace=True)
df.shape
df.to_pickle(f'{outdir}codes_df.pkl')
