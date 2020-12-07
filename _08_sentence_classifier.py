import os
import re
import sys
import numpy as np
import pandas as pd
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

datadir = "/Users/martijac/Documents/Frailty/frailty_classifier/output/"

# load SENTENCES
# check for .csv in filename to avoid the .DSstore file
# load the notes from 2018
notes_2018 = [i for i in os.listdir(datadir + "notes_labeled_embedded_SENTENCES/")
              if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{datadir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
        cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
#generate 'note' label (used in webanno and notes_labeled_embedded)
uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
# conc_notes_df contains official list of eligible patients
notes_2018_in_cndf = [i for i in notes_2018 if
                      "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if
                  "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
# get notes_labeled_embedded that match eligible patients only
df = pd.concat([pd.read_csv(datadir + "notes_labeled_embedded_SENTENCES/" + i) for i in
                notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)
# reset the index
df2 = df.reset_index()

# set a unique sentence id
sentence = []
sent = -1
for s in range(df2.shape[0]):
    if df2.iloc[s]['sentence'] != df2.iloc[s - 1]['sentence']:
        sent += 1
    sentence.append(sent)
df2['sentence_id'] = sentence

#dummies for labels
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
y_dums = pd.concat(
    [pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
cols = list(['note', 'sentence', 'token']) + list(y_dums.columns)
df2 = pd.concat([y_dums, df2], axis=1)

#label each sentence using heirachical rule:
# Positive label if any token is positive
# Negative label if there are no positive tokens and any token is negative

# aggregate tags by sentence
df2_label = df2.groupby('sentence_id', as_index=False).agg(
    n_tokens=('token', 'count'),
    any_Msk_prob_neg=('Msk_prob_-1', max),
    Msk_prob_pos=('Msk_prob_1', max),
    any_Nutrition_neg=('Nutrition_-1', max),
    Nutrition_pos=('Nutrition_1', max),
    any_Resp_imp_neg=('Resp_imp_-1', max),
    Resp_imp_pos=('Resp_imp_1', max),
    any_Fall_risk_neg=('Fall_risk_-1', max),
    Fall_risk_pos=('Fall_risk_1', max),
)
#add negative & neutral label using heirarchical rule
for n in out_varnames:
    df2_label[f"{n}_neg"] = np.where(((df2_label[f"{n}_pos"] != 1) & (df2_label[f"any_{n}_neg"] == 1)), 1, 0)
    df2_label[f"{n}_neut"] = np.where(((df2_label[f"{n}_pos"] != 1) & (df2_label[f"{n}_neg"] != 1)), 1, 0)
#drop extra columns
df2_label = df2_label.loc[:, ~df2_label.columns.str.startswith('any_')].copy()

# make empty df
clmns = ['sentence_id']
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    clmns.append(f"min_{v}")
    clmns.append(f"max_{v}")
    clmns.append(f"mean_{v}")
embeddings = pd.DataFrame(0, index=range(df2.sentence_id.nunique()), columns=clmns)
embeddings['sentence_id'] = list(df2.sentence_id.drop_duplicates())
# for each sentence, find the element-wise min/max/mean for embeddings
for v in range(0, df2.columns.str.startswith('identity_').sum()):
    embeddings[f"min_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg(min)
    embeddings[f"max_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg(max)
    embeddings[f"mean_{v}"] = df2.groupby('sentence_id')[f"identity_{v}"].agg('mean')
# check
# embeddings[embeddings.sentence_id == 100][['sentence_id', 'min_25', 'max_25', 'mean_15']]
# df2[df2.sentence_id == 100][['sentence_id', 'identity_25']]


#combine labels with embeddings
sentence_label_embed = df2_label.join(embeddings.set_index('sentence_id'), on='sentence_id')

#make output for RF/glmnet (from window_classifier_alt)