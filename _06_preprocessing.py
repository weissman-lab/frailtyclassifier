
'''
Takes the output of _05_ingest_to_sentence.py and does the following:
1.) Concatenate notes that are eligible for training/validation
2.) Label each sentence using the "heirarchical rule"
3.) For each sentence, get the labels, structured data, and element-wise
 min/max/mean for the embeddings.
4.) Split the data into 10 folds for cross validation
5.) For each fold, get the output labels, PCA of structured data, caseweights,
embeddings and 300- & 1000-d SVD of tf-idf
'''

import os
import re
from configargparse import ArgParser
import numpy as np
import pandas as pd
import copy
import hashlib
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from _99_project_module import send_message_to_slack, write_pickle

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass

def hasher(x):
    hash_object = hashlib.sha512(x.encode('utf-8'))
    hex_dig = hash_object.hexdigest()    
    return hex_dig


def compselect(d, pct, seed=0):
    '''function returns the number of components accounting for `pct`% of variance'''
    ncomp = d.shape[1]//2
    while True:
        pca = TruncatedSVD(n_components=ncomp, random_state=seed)
        pca.fit(d)
        cus = pca.explained_variance_ratio_.sum()
        if cus < pct:
            ncomp += ncomp//2
        else:
            ncomp = np.where((pca.explained_variance_ratio_.cumsum()>pct) == True)[0].min()
            return ncomp


def main():
    try:
        p = ArgParser()
        p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
        options = p.parse_args()
        batchstring = options.batchstring
        # identify the active learning directory
        outdir = f"{os.getcwd()}/output/"
        ALdir = f"{outdir}/saved_models/AL{batchstring}"
        sheepish_mkdir(ALdir)
        # make files for the CSV output
        sheepish_mkdir(f"{ALdir}/processed_data/")
        sheepish_mkdir(f"{ALdir}/processed_data/svd")
        sheepish_mkdir(f"{ALdir}/processed_data/embeddings")
        sheepish_mkdir(f"{ALdir}/processed_data/trvadata")
        sheepish_mkdir(f"{ALdir}/processed_data/caseweights")
        sheepish_mkdir(f"{ALdir}/processed_data/sklearn_artifacts")
        notes_2018 = [i for i in
                  os.listdir(outdir + "notes_labeled_embedded_SENTENCES/")
                  if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
        cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
        cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
        cndf['month'] = cndf.LATEST_TIME.dt.month + (
                cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
        # generate 'note' label (used in webanno and notes_labeled_embedded)
        cndf.month = cndf.month.astype(str)
        uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
        # conc_notes_df contains official list of eligible patients
        notes_2018_in_cndf = [i for i in notes_2018 if
                              "_".join(i.split("_")[-2:]) in uidstr]
        notes_excluded = [i for i in notes_2018 if
                          "_".join(i.split("_")[-2:]) not in uidstr]
        assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
        # remove double-annotated notes
        '''
        25 Jan 2021:  removing double-annotated notes, and notes from different 
        span of same patient.  Move the actual files so they don't get picked up later.
        As such, the code below will only do culling once.
        It will also move notes that were excluded in the July 2020 cull
        '''
        pids = set([re.sub(".csv", "", i.split("_")[-1]) for i in notes_2018_in_cndf])
        keepers = []
        Nhashed = 0
        for i in pids:
            notes_i = [j for j in notes_2018_in_cndf if i in j]
            if len(notes_i) >1:
                # check and see if there are notes from different batches.  Use the last one if so
                batchstrings = [k.split("_")[1] for k in notes_i]
                assert all(["AL" in k for k in batchstrings]), "not all double-coded notes are from an AL round."
                in_latest_batch = [k for k in notes_i if max(batchstrings) in k]
                # deal with a couple of manual cases
                if hasher(i) == 'faef3f1f1a76c57e42f9b35a662656096b4e2dfe15040a61a896b1de06ef1e0a45e61e7e9b26f9282047847854d2d1887d19cbf3041aff2130e102d65243e724':
                    keepers.append(f'enote_AL01_v2_m2_{i}.csv')
                    print(i)
                    Nhashed +=1
                elif hasher(i) == 'e13eced415697e4f59bcc8e75659dcffa4182a8de44b976d5e1d8160407711d276e38946ec52ec366a3ad92f197d92e8d56c1bd4e9029103d17ac12944cc3bc5':
                    keepers.append(f'enote_AL01_m2_{i}.csv')
                    print(i)
                    Nhashed +=1
                elif len(in_latest_batch) == 1:
                    keepers.append(in_latest_batch[0])
                elif len(set([k.split("_")[-2] for k in in_latest_batch]))>1: # deal with different spans
                    spans = [k.split("_")[-2] for k in in_latest_batch]
                    latest_span = [k for k in in_latest_batch if max(spans) in k]
                    assert len(latest_span) == 1
                    keepers.append(latest_span[0])
                elif any(['v2' in k for k in in_latest_batch]): # deal with the case of the "v2" notes -- an outgrowth of confusion around the culling in July 2020
                    v2_over_v1 = [k for k in in_latest_batch if 'v2' in k]
                    assert len(v2_over_v1) == 1
                    keepers.append(v2_over_v1[0])
                else:
                    print('problem with culling')
                    breakpoint()
            else:
                keepers.append(notes_i[0])
        droppers = [i for i in notes_2018_in_cndf if i not in keepers]
        assert (Nhashed == 2) | (len(droppers) == 0)
        assert len(keepers) == len(pids)
        if any(droppers):
            msg = "The following notes are getting moved/dropped because of duplication:" + "\n".join(droppers) + f"\n\nthere are {len(droppers)} of them"
            send_message_to_slack(msg)
        enote_dir = f"{outdir}/notes_labeled_embedded_SENTENCES"
        drop_dir = f"{enote_dir}/dropped_notes"
        sheepish_mkdir(drop_dir)
        assert os.path.exists(enote_dir)
        assert os.path.exists(drop_dir)
        for i in droppers:
            fra = enote_dir + "/" + i
            till = drop_dir + "/" + i
            os.rename(fra, till)
        assert all([i in os.listdir(enote_dir) for i in keepers])
        assert all([i in os.listdir(drop_dir) for i in droppers])
        ##################
        # load the files
        df = pd.concat([pd.read_csv(f"{outdir}notes_labeled_embedded_SENTENCES/{i}", 
                                    index_col = 0,
                                    dtype=dict(PAT_ID=str)) for i in keepers])
        df = df.drop(columns = ['sent_start', 'length'])
        ###########
        # Load and process structured data
        strdat = pd.read_csv(f"{outdir}structured_data_merged_cleaned.csv", 
                             index_col = 0)    
        # set seed
        seed = 8675309
        str_varnames = ['n_encs', 'n_ed_visits', 'n_admissions', 'days_hospitalized',
                        'mean_sys_bp', 'mean_dia_bp', 'sd_sys_bp', 'sd_dia_bp',
                        'bmi_mean', 'bmi_slope', 'max_o2', 'spo2_worst', 'ALBUMIN',
                        'ALKALINE_PHOSPHATASE', 'AST', 'BILIRUBIN', 'BUN', 'CALCIUM',
                        'CO2', 'CREATININE', 'HEMATOCRIT', 'HEMOGLOBIN', 'LDL', 'MCHC',
                        'MCV', 'PLATELETS', 'POTASSIUM', 'PROTEIN', 'RDW', 'SODIUM',
                        'WBC', 'sd_ALBUMIN', 'sd_ALKALINE_PHOSPHATASE', 'sd_AST',
                        'sd_BILIRUBIN', 'sd_BUN', 'sd_CALCIUM', 'sd_CO2',
                        'sd_CREATININE', 'sd_HEMATOCRIT', 'sd_HEMOGLOBIN', 'sd_LDL',
                        'sd_MCHC', 'sd_MCV', 'sd_PLATELETS', 'sd_POTASSIUM',
                        'sd_PROTEIN', 'sd_RDW', 'sd_SODIUM', 'sd_WBC', 'n_ALBUMIN',
                        'n_ALKALINE_PHOSPHATASE', 'n_AST', 'n_BILIRUBIN', 'n_BUN',
                        'n_CALCIUM', 'n_CO2', 'n_CREATININE', 'n_HEMATOCRIT',
                        'n_HEMOGLOBIN', 'n_LDL', 'n_MCHC', 'n_MCV', 'n_PLATELETS',
                        'n_POTASSIUM', 'n_PROTEIN', 'n_RDW', 'n_SODIUM', 'n_WBC',
                        'FERRITIN', 'IRON', 'MAGNESIUM', 'TRANSFERRIN',
                        'TRANSFERRIN_SAT', 'sd_FERRITIN', 'sd_IRON', 'sd_MAGNESIUM',
                        'sd_TRANSFERRIN', 'sd_TRANSFERRIN_SAT', 'n_FERRITIN', 'n_IRON',
                        'n_MAGNESIUM', 'n_TRANSFERRIN', 'n_TRANSFERRIN_SAT', 'PT',
                        'sd_PT', 'n_PT', 'PHOSPHATE', 'sd_PHOSPHATE', 'n_PHOSPHATE',
                        'PTT', 'sd_PTT', 'n_PTT', 'TSH', 'sd_TSH', 'n_TSH',
                        'n_unique_meds', 'elixhauser', 'n_comorb', 'AGE', 'SEX_Female',
                        'SEX_Male', 'MARITAL_STATUS_Divorced', 'MARITAL_STATUS_Married',
                        'MARITAL_STATUS_Other', 'MARITAL_STATUS_Single',
                        'MARITAL_STATUS_Widowed', 'EMPY_STAT_Disabled',
                        'EMPY_STAT_Full Time', 'EMPY_STAT_Not Employed',
                        'EMPY_STAT_Other', 'EMPY_STAT_Part Time', 'EMPY_STAT_Retired',
                        'MV_n_encs', 'MV_n_ed_visits', 'MV_n_admissions',
                        'MV_days_hospitalized', 'MV_mean_sys_bp', 'MV_mean_dia_bp',
                        'MV_sd_sys_bp', 'MV_sd_dia_bp', 'MV_bmi_mean', 'MV_bmi_slope',
                        'MV_max_o2', 'MV_spo2_worst', 'MV_ALBUMIN',
                        'MV_ALKALINE_PHOSPHATASE', 'MV_AST', 'MV_BILIRUBIN', 'MV_BUN',
                        'MV_CALCIUM', 'MV_CO2', 'MV_CREATININE', 'MV_HEMATOCRIT',
                        'MV_HEMOGLOBIN', 'MV_LDL', 'MV_MCHC', 'MV_MCV', 'MV_PLATELETS',
                        'MV_POTASSIUM', 'MV_PROTEIN', 'MV_RDW', 'MV_SODIUM', 'MV_WBC',
                        'MV_sd_ALBUMIN', 'MV_sd_ALKALINE_PHOSPHATASE', 'MV_sd_AST',
                        'MV_sd_BILIRUBIN', 'MV_sd_BUN', 'MV_sd_CALCIUM', 'MV_sd_CO2',
                        'MV_sd_CREATININE', 'MV_sd_HEMATOCRIT', 'MV_sd_HEMOGLOBIN',
                        'MV_sd_LDL', 'MV_sd_MCHC', 'MV_sd_MCV', 'MV_sd_PLATELETS',
                        'MV_sd_POTASSIUM', 'MV_sd_PROTEIN', 'MV_sd_RDW',
                        'MV_sd_SODIUM', 'MV_sd_WBC', 'MV_n_ALBUMIN',
                        'MV_n_ALKALINE_PHOSPHATASE', 'MV_n_AST', 'MV_n_BILIRUBIN',
                        'MV_n_BUN', 'MV_n_CALCIUM', 'MV_n_CO2', 'MV_n_CREATININE',
                        'MV_n_HEMATOCRIT', 'MV_n_HEMOGLOBIN', 'MV_n_LDL', 'MV_n_MCHC',
                        'MV_n_MCV', 'MV_n_PLATELETS', 'MV_n_POTASSIUM', 'MV_n_PROTEIN',
                        'MV_n_RDW', 'MV_n_SODIUM', 'MV_n_WBC', 'MV_FERRITIN',
                        'MV_IRON', 'MV_MAGNESIUM', 'MV_TRANSFERRIN',
                        'MV_TRANSFERRIN_SAT', 'MV_sd_FERRITIN', 'MV_sd_IRON',
                        'MV_sd_MAGNESIUM', 'MV_sd_TRANSFERRIN',
                        'MV_sd_TRANSFERRIN_SAT', 'MV_n_FERRITIN', 'MV_n_IRON',
                        'MV_n_MAGNESIUM', 'MV_n_TRANSFERRIN', 'MV_n_TRANSFERRIN_SAT',
                        'MV_PT', 'MV_sd_PT', 'MV_n_PT', 'MV_PHOSPHATE',
                        'MV_sd_PHOSPHATE', 'MV_n_PHOSPHATE', 'MV_PTT', 'MV_sd_PTT',
                        'MV_n_PTT', 'MV_TSH', 'MV_sd_TSH', 'MV_n_TSH',
                        'MV_n_unique_meds', 'MV_elixhauser', 'MV_n_comorb', 'MV_AGE',
                        'MV_SEX', 'MV_MARITAL_STATUS', 'MV_EMPY_STAT']
        out_varnames = ['Msk_prob', 'Nutrition', 'Resp_imp', 'Fall_risk']
    
        strdat = strdat.drop(columns = ['RACE', 'LANGUAGE', 'MV_RACE', 'MV_LANGUAGE'])
        strdat = strdat.merge(df[['PAT_ID', 'month']].drop_duplicates())    
    
        # set a unique sentence id that does not reset to 0 with each note
        df.insert(2, 'sentence_id', df.note+"_sent"+df.sentence.astype(str))
        df.rename(columns={'sentence': 'sentence_in_note'}, inplace=True)
        # dummies for labels
        out_varnames = df.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
        y_dums = pd.concat(
            [pd.get_dummies(df[[i]].astype(str)) for i in out_varnames], axis=1)
        df = pd.concat([y_dums, df], axis=1)
        # label each sentence using heirachical rule:
        # Positive label if any token is positive
        # Negative label if there are no positive tokens and any token is negative
        df_label = df.groupby('sentence_id', as_index=True).agg(
            sentence=('token', lambda x: ' '.join(x.astype(str))),  # sentence tokens
            sentence_in_note=('sentence_in_note', 'first'),
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
        # need to set the group as the index, then reset it (unclear why this is not
        # equivalent to 'as_index=False')
        df_label = df_label.reset_index(drop=False)
        # add negative & neutral label using heirarchical rule
        for n in out_varnames:
            df_label[f"{n}_neg"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"any_{n}_neg"] == 1)), 1,
                0)
            df_label[f"{n}_neut"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"{n}_neg"] != 1)), 1, 0)
        # drop extra columns
        df_label = df_label.loc[:, ~df_label.columns.str.startswith('any_')].copy()
        df_label.insert(loc = 0, column = 'PAT_ID', value = df_label.sentence_id.apply(lambda x: x.split("_")[-2]))
        assert all([i in list(pids) for i in list(df_label.PAT_ID.unique())])
        assert all([i in list(df_label.PAT_ID.unique()) for i in list(pids)])
        # summarize embeddings (element-wise min/max/mean) for each sentence
        # first, make empty df
        
        clmns = ['sentence_id', 'note']
        for v in range(0, df.columns.str.startswith('identity_').sum()):
            clmns.append(f"min_{v}")
            clmns.append(f"max_{v}")
            clmns.append(f"mean_{v}")
        embeddings = pd.DataFrame(0, index=range(df.sentence_id.nunique()),
                                  columns=clmns)
        embeddings['sentence_id'] = list(df.sentence_id.drop_duplicates())
        embeddings['note'] = df.groupby('sentence_id', as_index=False)['note'].agg('first')['note']
        # for each sentence, find the element-wise min/max/mean for embeddings
        for v in range(0, df.columns.str.startswith('identity_').sum()):
            embeddings[f"min_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
                min)[f"identity_{v}"]
            embeddings[f"max_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
                max)[f"identity_{v}"]
            embeddings[f"mean_{v}"] = df.groupby('sentence_id', as_index=False)[f"identity_{v}"].agg(
                'mean')[f"identity_{v}"]

        # drop embeddings for center word
        embeddings2 = embeddings.loc[:,
                      ~embeddings.columns.str.startswith('identity')].copy()
        embeddings2.insert(loc = 2, column = 'PAT_ID', value = embeddings2.note.apply(lambda x: x.split("_")[-1]))
        assert all([i in list(pids) for i in list(embeddings2.PAT_ID.unique())])
        assert all([i in list(embeddings2.PAT_ID.unique()) for i in list(pids)])
        #############################
        # breaking out input files by fold
        seed = 8675309
        np.random.seed(seed)
        pids = list(pids)
        pids.sort()
        folds = [i % 10 for i in range(len(pids))]
        fold_definition = pd.DataFrame(dict(PAT_ID = pids)) # start a data frame for documenting which notes are in each fold
        fold_sklearn_artifacts = {}
        for repeat in [1,2,3]:
            fold_sklearn_artifacts[repeat] = {} # first level of dict is for repeats
            folds = np.random.choice(folds, len(folds), replace = False)
            fold_definition[f"repeat{repeat}"] = folds        
            fold_sklearn_artifacts[repeat] = {}
            for fold in range(10):
                print(f"starting fold {fold}, repeat {repeat}")
                tr = [pids[i] for i, j in enumerate(folds) if j != fold]
                va = [pids[i] for i, j in enumerate(folds) if j == fold]
                if (df.loc[df.PAT_ID.isin(va), out_varnames]!=0).sum().min() == 0:
                    print((df.loc[df.PAT_ID.isin(va), out_varnames]!=0).sum())
                    send_message_to_slack(f"Zeros at r{repeat} f{fold}")
                # PCA for structured data.  Scale, then fit PCA, then scale output
                scaler_in = StandardScaler()
                tr_scaled = scaler_in.fit_transform(strdat.loc[strdat.PAT_ID.isin(tr), str_varnames])
                tr_scaled = np.nan_to_num(tr_scaled) # set nans from constant columns to zero.  the PCA will drop them.  This is more robust than pre-removal, as it's more future-proof
                va_scaled = scaler_in.transform(strdat.loc[strdat.PAT_ID.isin(va), str_varnames])
                va_scaled = np.nan_to_num(va_scaled)
                ncomp = compselect(tr_scaled, .95)
                pca = TruncatedSVD(n_components=ncomp, random_state=seed)
                tr_rot = pca.fit_transform(tr_scaled)
                va_rot = pca.transform(va_scaled)
                scaler_out = StandardScaler()
                tr_rot_scaled = scaler_out.fit_transform(tr_rot)
                va_rot_scaled = scaler_out.transform(va_rot)
                fold_sklearn_artifacts[repeat][fold] = {}
                fold_sklearn_artifacts[repeat][fold]['scaler_in'] = scaler_in
                fold_sklearn_artifacts[repeat][fold]['pca'] = pca
                fold_sklearn_artifacts[repeat][fold]['scaler_out'] = scaler_out
                # data frames for merging
                print("A")
                str_tr = pd.concat([pd.DataFrame(dict(PAT_ID = tr)), 
                                    pd.DataFrame(tr_rot_scaled,
                                                 columns = ['pca' + str(i) for i in range(ncomp)])], 
                                   axis = 1)
                str_va = pd.concat([pd.DataFrame(dict(PAT_ID = va)), 
                                    pd.DataFrame(va_rot_scaled,
                                                 columns = ['pca' + str(i) for i in range(ncomp)])], 
                                   axis = 1)
                dflab = copy.deepcopy(df_label)
                # add case weights into the main dflab
                for i in out_varnames:
                    prevalence_tr = (dflab.loc[dflab.PAT_ID.isin(tr), f"{i}_neut"] ==0).mean()
                    val = (dflab[f"{i}_neut"] ==0)*1/prevalence_tr + (dflab[f"{i}_neut"] !=0)*1/(1-prevalence_tr)
                    val = val/val.mean()
                    val.value_counts()
                    assert (sum(val) - len(val))<.01
                    dflab.insert(loc = 0, column = f'{i}_cw', value = val)
                            
                cwdf = dflab.loc[dflab.PAT_ID.isin(tr), ['sentence_id']+[i+"_cw" for i in out_varnames]]
                                
                # merge the structured data onto the main df, and cut into training and test sets
                df_tr = dflab.loc[dflab.PAT_ID.isin(tr)].merge(str_tr)
                assert df_tr.shape[0] == dflab.loc[dflab.PAT_ID.isin(tr)].shape[0]
                df_va = dflab.loc[dflab.PAT_ID.isin(va)].merge(str_va)
                assert df_va.shape[0] == dflab.loc[dflab.PAT_ID.isin(va)].shape[0]
    
                print("B")
                # get embeddings for fold
                embeddings_tr = embeddings2[~embeddings2.PAT_ID.isin(tr)].reset_index(drop=True)
                embeddings_va = embeddings2[embeddings2.PAT_ID.isin(va)].reset_index(drop=True)
    
                # Convert text into matrix of tf-idf features:
                # id documents
                tr_docs = dflab.loc[dflab.PAT_ID.isin(tr), 'sentence'].tolist()
                # instantiate countvectorizer (turn off default stopwords)
                cv = CountVectorizer(analyzer='word', stop_words=None)
                # compute tf
                f_tr_tf = cv.fit_transform(tr_docs)
                # id additional stopwords: medlist_was_here_but_got_cut,
                # meds_was_here_but_got_cut, catv2_was_here_but_got_cut
                cuttext = '_was_here_but_got_cut'
                stopw = [i for i in list(cv.get_feature_names()) if re.search(cuttext, i)]
                # repeat countvec with full list of stopwords
                cv = CountVectorizer(analyzer='word', stop_words=stopw)
                # fit to data, then transform to count matrix
                f_tr_tf = cv.fit_transform(tr_docs)
                # fit to count matrix, then transform to tf-idf representation
                tfidf_transformer = TfidfTransformer()
                f_tr_tfidf = tfidf_transformer.fit_transform(f_tr_tf)
                # apply feature extraction to test set (do NOT fit on test data)
                va_docs = dflab.loc[dflab.PAT_ID.isin(va), 'sentence'].tolist()
                f_va_tf = cv.transform(va_docs)
                f_va_tfidf = tfidf_transformer.transform(f_va_tf)
                # dimensionality reduction with truncated SVD
                svd_300 = TruncatedSVD(n_components=300, n_iter=5, random_state=9082020)
                svd_1000 = TruncatedSVD(n_components=1000, n_iter=5, random_state=9082020)
                # fit to training data & transform
                f_tr_svd300 = pd.DataFrame(svd_300.fit_transform(f_tr_tfidf))
                f_tr_svd1000 = pd.DataFrame(svd_1000.fit_transform(f_tr_tfidf))
                # transform test data (do NOT fit on test data)
                f_va_svd300 = pd.DataFrame(svd_300.transform(f_va_tfidf))
                f_va_svd1000 = pd.DataFrame(svd_1000.transform(f_va_tfidf))
                # sklearn stuff
                fold_sklearn_artifacts[repeat][fold]['cv'] = cv
                fold_sklearn_artifacts[repeat][fold]['tfidf_transformer'] = tfidf_transformer
                fold_sklearn_artifacts[repeat][fold]['svd_300'] = svd_300
                fold_sklearn_artifacts[repeat][fold]['svd_1000'] = svd_1000
                # Output for r
                df_tr.to_csv(f"{ALdir}/processed_data/trvadata/r{repeat}_f{fold}_tr_df.csv")
                df_va.to_csv(f"{ALdir}/processed_data/trvadata/r{repeat}_f{fold}_va_df.csv")
                # f_tr_cw.to_csv(f"{trvadatadir}r{r + 1}_f{f + 1}_tr_cw.csv")
                embeddings_tr.to_csv(f"{ALdir}/processed_data/embeddings/r{repeat}_f{fold}_tr_embed_min_max_mean_SENT.csv")
                embeddings_va.to_csv(f"{ALdir}/processed_data/embeddings/r{repeat}_f{fold}_va_embed_min_max_mean_SENT.csv")
                f_tr_svd300.to_csv(f"{ALdir}/processed_data/svd/r{repeat}_f{fold}_tr_svd300.csv")
                f_tr_svd1000.to_csv(f"{ALdir}/processed_data/svd/r{repeat}_f{fold}_tr_svd1000.csv")
                f_va_svd300.to_csv(f"{ALdir}/processed_data/svd/r{repeat}_f{fold}_va_svd300.csv")
                f_va_svd1000.to_csv(f"{ALdir}/processed_data/svd/r{repeat}_f{fold}_va_svd1000.csv")
                # case weights
                cwdf.to_csv(f"{ALdir}/processed_data/caseweights/r{repeat}_f{fold}_tr_caseweights.csv")
                write_pickle(fold_sklearn_artifacts, f"{ALdir}/processed_data/sklearn_artifacts/r{repeat}_f{fold}skl_dict.pkl")
        fold_definition.to_csv(f"{ALdir}/processed_data/fold_definitions.csv")
    except Exception as e:
        print(e)
        breakpoint()

if __name__ == "__main__":
    main()




