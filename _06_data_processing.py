from configargparse import ArgParser
import pandas as pd
import numpy as np
import os
import re
import copy
from sklearn.impute import SimpleImputer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfTransformer
from utils.constants import OUT_VARNAMES, STR_VARNAMES
from utils.organization import summarize_train_test_split
from utils.misc import (sheepish_mkdir, write_pickle, read_pickle, hasher,
                        send_message_to_slack, write_txt, compselect)


class DataProcessor():
    def __init__(self, batchstring):
        self.outdir = f"{os.getcwd()}/output/"
        self.ALdir = f"{self.outdir}/saved_models/AL{batchstring}"
        self.batchstring = batchstring
        sheepish_mkdir(self.ALdir)
        # make files for the CSV output
        sheepish_mkdir(f"{self.ALdir}/processed_data/")
        sheepish_mkdir(f"{self.ALdir}/processed_data/svd")
        sheepish_mkdir(f"{self.ALdir}/processed_data/embeddings")
        sheepish_mkdir(f"{self.ALdir}/processed_data/trvadata")
        sheepish_mkdir(f"{self.ALdir}/processed_data/full_set")
        sheepish_mkdir(f"{self.ALdir}/processed_data/full_set_earlystopping")
        sheepish_mkdir(f"{self.ALdir}/processed_data/caseweights")
        sheepish_mkdir(f"{self.ALdir}/processed_data/sklearn_artifacts")
        summarize_train_test_split()
        if "data_dict.pkl" not in os.listdir(f"{self.ALdir}/processed_data/"):
            self.data_dict = self.load_clean_aggregate()
            write_pickle(self.data_dict, f"{self.ALdir}/processed_data/data_dict.pkl")
        else:
            self.data_dict = read_pickle(f"{self.ALdir}/processed_data/data_dict.pkl")
        if 'fold_definition.csv' not in os.listdir(f"{self.ALdir}/processed_data/"):
            try:
                self.fold_definition = self.establish_folds()
            except RecursionError:
                print("Could not construct 10 folds with all class labels!")
                self.fold_definition = self.establish_folds(recurse = False)
            self.fold_definition.to_csv(f"{self.ALdir}/processed_data/fold_definition.csv")
        else:
            self.fold_definition = pd.read_csv(f"{self.ALdir}/processed_data/fold_definition.csv", index_col=0)
        # defined in methods
        self.tr_pids = None

    def establish_folds(self, seed=0, recurse=True):
        # this function ensures folds all have representation from each class
        pids = self.data_dict['pids']
        df_label = self.data_dict['df_label']
        np.random.seed(seed)
        folds = [i % 10 for i in range(len(pids))]
        fold_definition = pd.DataFrame(
            dict(PAT_ID=pids))  # start a data frame for documenting which notes are in each fold
        for repeat in [1, 2, 3]:
            folds = np.random.choice(folds, len(folds), replace=False)
            fold_definition[f"repeat{repeat}"] = folds
            for fold in range(10):
                # print(f"starting fold {fold}, repeat {repeat}")
                tr = [pids[i] for i, j in enumerate(folds) if j != fold]
                va = [pids[i] for i, j in enumerate(folds) if j == fold]
                if any(df_label.loc[df_label.PAT_ID.isin(va)].nunique() == 1) | any(
                        df_label.loc[df_label.PAT_ID.isin(tr)].nunique() == 1):
                    print(f"Seed {seed} didn't work.  Incrementing.")
                    if recurse:
                        return self.establish_folds(seed + 1)
        print(f'Worked at seed {seed}')
        return fold_definition

    def load_clean_aggregate(self):
        '''
        This method will use the batchstring and the file
        `./output/notes_labeled_embedded_SENTENCES/notes_train_official.csv`
        to subset the notes to notes that were added to the pipeline before the batch indicated in the batchstring
        '''
        notes_2018 = [i for i in
                      os.listdir(self.outdir + "notes_labeled_embedded_SENTENCES/")
                      if '.csv' in i and "enote" in i and int(i.split("_")[-2][1:]) < 13]
        cndf = pd.read_pickle(f"{self.outdir}conc_notes_df.pkl")
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
        notes_2018_in_cndf.sort()
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
            if len(notes_i) > 1:
                # check and see if there are notes from different batches.  Use the last one if so
                batchstrings = [k.split("_")[1] for k in notes_i]
                assert all(["AL" in k for k in batchstrings]), "not all double-coded notes are from an AL round."
                in_latest_batch = [k for k in notes_i if max(batchstrings) in k]
                # deal with a couple of manual cases
                if hasher(
                        i) == 'faef3f1f1a76c57e42f9b35a662656096b4e2dfe15040a61a896b1de06ef1e0a45e61e7e9b26f9282047847854d2d1887d19cbf3041aff2130e102d65243e724':
                    keepers.append(f'enote_AL01_v2_m2_{i}.csv')
                    print(i)
                    Nhashed += 1
                elif hasher(
                        i) == 'e13eced415697e4f59bcc8e75659dcffa4182a8de44b976d5e1d8160407711d276e38946ec52ec366a3ad92f197d92e8d56c1bd4e9029103d17ac12944cc3bc5':
                    keepers.append(f'enote_AL01_m2_{i}.csv')
                    print(i)
                    Nhashed += 1
                elif len(in_latest_batch) == 1:
                    keepers.append(in_latest_batch[0])
                elif len(set([k.split("_")[-2] for k in in_latest_batch])) > 1:  # deal with different spans
                    spans = [k.split("_")[-2] for k in in_latest_batch]
                    latest_span = [k for k in in_latest_batch if max(spans) in k]
                    assert len(latest_span) == 1
                    keepers.append(latest_span[0])
                elif any(['v2' in k for k in
                          in_latest_batch]):  # deal with the case of the "v2" notes -- an outgrowth of confusion around the culling in July 2020
                    v2_over_v1 = [k for k in in_latest_batch if 'v2' in k]
                    assert len(v2_over_v1) == 1
                    keepers.append(v2_over_v1[0])
                else:
                    print('problem with culling')
                    breakpoint()
            else:
                keepers.append(notes_i[0])
        droppers = [i for i in notes_2018_in_cndf if i not in keepers]
        # make a little report on keepers and droppers
        report = 'Keepers:\n'
        for i in range(int(self.batchstring)):
            x = [j for j in keepers if f'AL0{str(i)}' in j]
            report += f"{len(x)} notes in AL0{i}\n"
        x = [i for i in keepers if "AL0" not in i]
        report += f"{len(x)} notes from initial random batches\n"
        report += f"Total of {len(keepers)} notes keeping\n\n"
        report += "Droppers\n"
        for i in range(int(self.batchstring)):
            x = [j for j in droppers if f'AL0{str(i)}' in j]
            report += f"{len(x)} notes in AL0{i}\n"
        x = [i for i in droppers if "AL0" not in i]
        report += f"{len(x)} notes from initial random batches\n"
        report += f"Total of {len(droppers)} notes dropping"
        report += "\n\nHere are the notes that got dropped, and their corresponding keeper:\n-----------------\n"
        for i in droppers:
            p = re.sub(".csv", "", i.split("_")[-1])
            k = [j for j in keepers if p in j]
            report += f"Dropping: {i}\n"
            report += f"Keeping: {k[0]}\n---------\n"
        print(report)
        write_txt(report, f"{self.ALdir}/keep_drop_report.txt")
        assert (Nhashed == 2) | (len(droppers) == 0)
        assert len(keepers) == len(pids)
        if any(droppers):
            msg = "The following notes are getting moved/dropped because of duplication:" + "\n".join(
                droppers) + f"\n\nthere are {len(droppers)} of them"
            send_message_to_slack(msg)
        enote_dir = f"{self.outdir}/notes_labeled_embedded_SENTENCES"
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
        master_notelist = pd.read_csv('./output/notes_labeled_embedded_SENTENCES/notes_train_official.csv',
                                      index_col=0)
        assert all([i in master_notelist.filename.tolist() for i in keepers])
        assert all([i in keepers for i in master_notelist.filename.tolist()])

        # subset using batchstring
        subbatch = [i if int(re.sub("AL", "", i.split("_")[1])) < int(self.batchstring) else None for i in keepers if
                    "_AL" in i]
        subbatch = [i for i in subbatch if i is not None]
        subbatch += [i for i in keepers if "batch" in i]
        pids = set([re.sub(".csv", "", i.split("_")[-1]) for i in subbatch])

        ##################
        # load the files
        df = pd.concat([pd.read_csv(f"{self.outdir}notes_labeled_embedded_SENTENCES/{i}",
                                    index_col=0,
                                    dtype=dict(PAT_ID=str)) for i in subbatch])
        df = df.drop(columns=['sent_start', 'length'])
        ###########
        # Load and process structured data
        strdat = pd.read_csv(f"{self.outdir}structured_data_merged_cleaned.csv",
                             index_col=0)
        strdat = strdat.drop(columns=['RACE', 'LANGUAGE', 'MV_RACE', 'MV_LANGUAGE'])
        strdat = strdat.merge(df[['PAT_ID', 'month']].drop_duplicates())
        # set a unique sentence id that does not reset to 0 with each note
        # first prepend zeros to the sentence numbers so that the sorting works out
        sent_str = df.sentence.apply(lambda x: "".join(["0" for i in range(6 - len(str(x)))]) + str(x))
        df.insert(2, 'sentence_id', df.note + "_sent" + sent_str)
        df.rename(columns={'sentence': 'sentence_in_note'}, inplace=True)
        # dummies for labels
        y_dums = pd.concat(
            [pd.get_dummies(df[[i]].astype(str)) for i in OUT_VARNAMES], axis=1)
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
        for n in OUT_VARNAMES:
            df_label[f"{n}_neg"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"any_{n}_neg"] == 1)), 1,
                0)
            df_label[f"{n}_neut"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"{n}_neg"] != 1)), 1, 0)
        # drop extra columns
        df_label = df_label.loc[:, ~df_label.columns.str.startswith('any_')].copy()
        df_label.insert(loc=0, column='PAT_ID', value=df_label.sentence_id.apply(lambda x: x.split("_")[-2]))
        assert all([i in list(pids) for i in list(df_label.PAT_ID.unique())])
        assert all([i in list(df_label.PAT_ID.unique()) for i in list(pids)])
        # summarize embeddings (element-wise min/max/mean) for each sentence
        mu = df.groupby(['PAT_ID', 'sentence_id'])[[i for i in df.columns if 'identity' in i]].mean().reset_index()
        mx = df.groupby(['PAT_ID', 'sentence_id'])[[i for i in df.columns if 'identity' in i]].max().reset_index()
        mn = df.groupby(['PAT_ID', 'sentence_id'])[[i for i in df.columns if 'identity' in i]].min().reset_index()
        mu = mu.rename(columns={i: re.sub('identity', 'mean', i) for i in mu.columns})
        mx = mx.rename(columns={i: re.sub('identity', 'max', i) for i in mx.columns})
        mn = mn.rename(columns={i: re.sub('identity', 'min', i) for i in mn.columns})
        emb = mu.merge(mx.merge(mn))
        assert emb.shape[0] == mu.shape[0]
        assert emb.shape[1] == 902  # 300D embeddings
        assert all([i in list(pids) for i in list(emb.PAT_ID.unique())])
        assert all([i in list(emb.PAT_ID.unique()) for i in list(pids)])
        assert all(df_label.sentence_id.isin(emb.sentence_id))
        assert all(emb.sentence_id.isin(df_label.sentence_id))

        # # generate summary stats by batch
        # # for each batch, compute the mean, sd, quartiles of variable
        # # first summarize the label data for merging onto the structured data
        # sumlab = copy.deepcopy(df_label)
        # sumlab['batch'] = sumlab.sentence_id.apply(lambda x: x.split('_')[0])
        # sumlab['month'] = sumlab.sentence_id.apply(lambda x: re.sub("ALTERNATE", "", x.split('_')[1][1:])).astype(int)

        # sumlab.month.value_counts()
        # lab_sum = sumlab.groupby(['PAT_ID', 'batch'])[[i for i in sumlab.columns if i not in ['sentence_id', 'sentence', 'sentence_in_note']]].sum().reset_index()
        # sumdat = lab_sum.merge(sumdat, how = 'outer')

        # ssrows = []
        # b = 'AL00'
        # sumdat.loc[sumdat.batch == b].shape

        pids = list(pids)
        pids.sort()

        return dict(strdat=strdat,
                    df_label=df_label,
                    emb=emb,
                    pids=pids)

    def stratify_trva(self):
        '''
        Function picks a random seed that is maximally balanced between the training and validaiton set labels
        It tries out several random seeds, and picks the one that minimizes the mean squared deviation in label
        prevalence across labels
        '''
        savepath = f"{self.ALdir}/processed_data/full_set_earlystopping/trva_split.pkl"
        df = self.data_dict['df_label']
        if os.path.exists(savepath):
            self.tr_pids = read_pickle(savepath)['training_pids']
        else:
            seedres = []
            for seed in range(1000):
                np.random.seed(seed)
                tr = np.random.choice(a = df.PAT_ID.unique(), size = int(df.PAT_ID.nunique()*.8), replace = False)
                labs = [i for i in df.columns if any([j for j in OUT_VARNAMES if j in i and "_cw" not in i])]
                tr_prev = df.loc[df.PAT_ID.isin(tr), labs]
                va_prev = df.loc[~df.PAT_ID.isin(tr), labs]
                msd = ((tr_prev.mean() - va_prev.mean())**2).mean()
                seedres.append(dict(seed = seed, msd = msd))
            seeddf = pd.DataFrame(seedres)
            best_seed = seeddf.loc[seeddf.msd == seeddf.msd.min(), 'seed'].iloc[0]
            np.random.seed(best_seed)
            tr = np.random.choice(a=df.PAT_ID.unique(), size=int(df.PAT_ID.nunique() * .8), replace=False)
            tosave = dict(seed = best_seed,
                          training_pids = list(tr))
            write_pickle(tosave, savepath)
            self.tr_pids = list(tr)

    def process_fold(self,
                     fold,
                     repeat,
                     final_80_20 = False):
        # dump some variables from the data dict into local scope for easier code
        strdat = self.data_dict['strdat']
        emb = self.data_dict['emb']
        df_label = self.data_dict['df_label']
        if final_80_20 == False:
            tr = self.fold_definition.loc[self.fold_definition[f'repeat{repeat}'] != fold, 'PAT_ID'].tolist()
            va = self.fold_definition.loc[self.fold_definition[f'repeat{repeat}'] == fold, 'PAT_ID'].tolist()
            print("here is the validation set:")
            print(self.fold_definition.loc[self.fold_definition[f'repeat{repeat}'] == fold])
            outloc = 'trvadata' # the final dave path will be ALdir/processed_data/outloc
        else:
            self.stratify_trva() # dump the training PIDs into self.tr_pids
            tr = self.tr_pids
            va = list(self.data_dict['df_label']\
                      .loc[~self.data_dict['df_label'].PAT_ID.isin(self.tr_pids), 'PAT_ID'].unique())
            outloc = 'full_set_earlystopping' # the final dave path will be ALdir/processed_data/outloc
        # impute
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        strdat_imp_tr = imputer.fit_transform(strdat.loc[strdat.PAT_ID.isin(tr), STR_VARNAMES])
        strdat_imp_va = imputer.transform(strdat.loc[strdat.PAT_ID.isin(va), STR_VARNAMES])
        # PCA for structured data.  Scale, then fit PCA, then scale output
        scaler_in = StandardScaler()
        tr_scaled = scaler_in.fit_transform(strdat_imp_tr)
        va_scaled = scaler_in.transform(strdat_imp_va)
        ncomp = compselect(tr_scaled, .95)
        pca = TruncatedSVD(n_components=ncomp, random_state=8675309)
        tr_rot = pca.fit_transform(tr_scaled)
        va_rot = pca.transform(va_scaled)
        scaler_out = StandardScaler()
        tr_rot_scaled = scaler_out.fit_transform(tr_rot)
        va_rot_scaled = scaler_out.transform(va_rot)
        sklearn_dict = {}
        sklearn_dict['imputer'] = imputer
        sklearn_dict['scaler_in'] = scaler_in
        sklearn_dict['pca'] = pca
        sklearn_dict['scaler_out'] = scaler_out
        # data frames for merging
        str_tr = pd.concat([pd.DataFrame(dict(PAT_ID=tr)),
                            pd.DataFrame(tr_rot_scaled,
                                         columns=['pca' + str(i) for i in range(ncomp)])],
                           axis=1)
        str_va = pd.concat([pd.DataFrame(dict(PAT_ID=va)),
                            pd.DataFrame(va_rot_scaled,
                                         columns=['pca' + str(i) for i in range(ncomp)])],
                           axis=1)
        dflab = copy.deepcopy(df_label)
        # add case weights into the main dflab
        for i in OUT_VARNAMES:
            prevalence_tr = (dflab.loc[dflab.PAT_ID.isin(tr), f"{i}_neut"] == 0).mean()
            val = (dflab[f"{i}_neut"] == 0) * 1 / prevalence_tr + (dflab[f"{i}_neut"] != 0) * 1 / (1 - prevalence_tr)
            val = val / val.mean()
            val.value_counts()
            assert (sum(val) - len(val)) < .01
            dflab.insert(loc=0, column=f'{i}_cw', value=val)
        cwdf = dflab.loc[dflab.PAT_ID.isin(tr), ['sentence_id'] + [i + "_cw" for i in OUT_VARNAMES]]
        # merge the structured data onto the main df, and cut into training and test sets
        df_tr = dflab.loc[dflab.PAT_ID.isin(tr)].merge(str_tr)
        assert df_tr.shape[0] == dflab.loc[dflab.PAT_ID.isin(tr)].shape[0]
        df_va = dflab.loc[dflab.PAT_ID.isin(va)].merge(str_va)
        assert df_va.shape[0] == dflab.loc[dflab.PAT_ID.isin(va)].shape[0]
        # get embeddings for fold
        embeddings_tr = emb[emb.PAT_ID.isin(tr)].reset_index(drop=True)
        embeddings_va = emb[emb.PAT_ID.isin(va)].reset_index(drop=True)
        assert df_tr.shape[0] == embeddings_tr.shape[0]
        assert df_va.shape[0] == embeddings_va.shape[0]
        # Convert text into matrix of tf-idf features:
        # id documents
        tr_docs = dflab.loc[dflab.PAT_ID.isin(tr), 'sentence'].tolist()
        tfidf_ids_tr = dflab.loc[dflab.PAT_ID.isin(tr), 'sentence_id'].tolist()
        tfidf_ids_va = dflab.loc[dflab.PAT_ID.isin(va), 'sentence_id'].tolist()
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
        # add sentence ID to all of the TFIDF data frames
        f_tr_svd300.insert(0, value=tfidf_ids_tr, column='sentence_id')
        f_tr_svd1000.insert(0, value=tfidf_ids_tr, column='sentence_id')
        f_va_svd300.insert(0, value=tfidf_ids_va, column='sentence_id')
        f_va_svd1000.insert(0, value=tfidf_ids_va, column='sentence_id')
        # sorting
        embeddings_tr = embeddings_tr.sort_values('sentence_id')
        embeddings_va = embeddings_va.sort_values('sentence_id')
        f_tr_svd300 = f_tr_svd300.sort_values('sentence_id')
        f_tr_svd1000 = f_tr_svd1000.sort_values('sentence_id')
        f_va_svd300 = f_va_svd300.sort_values('sentence_id')
        f_va_svd1000 = f_va_svd1000.sort_values('sentence_id')
        # checking
        assert all(df_tr.sentence_id.values == embeddings_tr.sentence_id.values)
        assert all(df_va.sentence_id.values == embeddings_va.sentence_id.values)
        assert all(df_tr.sentence_id.values == f_tr_svd300.sentence_id.values)
        assert all(df_va.sentence_id.values == f_va_svd300.sentence_id.values)
        assert all(df_tr.sentence_id.values == f_tr_svd1000.sentence_id.values)
        assert all(df_va.sentence_id.values == f_va_svd1000.sentence_id.values)
        # sklearn stuff
        sklearn_dict['cv'] = cv
        sklearn_dict['tfidf_transformer'] = tfidf_transformer
        sklearn_dict['svd_300'] = svd_300
        sklearn_dict['svd_1000'] = svd_1000
        # Output for r
        df_tr.to_csv(f"{self.ALdir}/processed_data/{outloc}/r{repeat}_f{fold}_tr_df.csv")
        df_va.to_csv(f"{self.ALdir}/processed_data/{outloc}/r{repeat}_f{fold}_va_df.csv")
        if outloc =='trvadata':
            embeddings_tr.to_csv(f"{self.ALdir}/processed_data/embeddings/r{repeat}_f{fold}_tr_embed_min_max_mean_SENT.csv")
            embeddings_va.to_csv(f"{self.ALdir}/processed_data/embeddings/r{repeat}_f{fold}_va_embed_min_max_mean_SENT.csv")
            f_tr_svd300.to_csv(f"{self.ALdir}/processed_data/svd/r{repeat}_f{fold}_tr_svd300.csv")
            f_tr_svd1000.to_csv(f"{self.ALdir}/processed_data/svd/r{repeat}_f{fold}_tr_svd1000.csv")
            f_va_svd300.to_csv(f"{self.ALdir}/processed_data/svd/r{repeat}_f{fold}_va_svd300.csv")
            f_va_svd1000.to_csv(f"{self.ALdir}/processed_data/svd/r{repeat}_f{fold}_va_svd1000.csv")
            # case weights
            cwdf.to_csv(f"{self.ALdir}/processed_data/caseweights/r{repeat}_f{fold}_tr_caseweights.csv")
            # sklearn artifacts
            write_pickle(sklearn_dict, f"{self.ALdir}/processed_data/sklearn_artifacts/r{repeat}_f{fold}sklearn_dict.pkl")
            print(f"Saved data from repeat {repeat}, fold {fold}")
        else:
            write_pickle(sklearn_dict, f"{self.ALdir}/processed_data/{outloc}/sklearn_dict.pkl")

    def process_full_training_set(self):
        strdat = self.data_dict['strdat']
        emb = self.data_dict['emb']
        df_label = self.data_dict['df_label']
        pids = self.data_dict['pids']
        imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        strdat_imp = imputer.fit_transform(strdat[STR_VARNAMES])
        scaler_in = StandardScaler()
        scaled = scaler_in.fit_transform(strdat_imp)
        ncomp = compselect(scaled, .95)
        pca = TruncatedSVD(n_components=ncomp, random_state=8675309)
        rot = pca.fit_transform(scaled)
        scaler_out = StandardScaler()
        rot_scaled = scaler_out.fit_transform(rot)
        sklearn_dict = {}
        sklearn_dict['imputer'] = imputer
        sklearn_dict['scaler_in'] = scaler_in
        sklearn_dict['pca'] = pca
        sklearn_dict['scaler_out'] = scaler_out
        str_all = pd.concat([pd.DataFrame(dict(PAT_ID=pids)),
                             pd.DataFrame(rot_scaled,
                                          columns=['pca' + str(i) for i in range(ncomp)])],
                            axis=1)
        dflab = copy.deepcopy(df_label)
        # add case weights into the main dflab
        for i in OUT_VARNAMES:
            prevalence = (dflab[f"{i}_neut"] == 0).mean()
            val = (dflab[f"{i}_neut"] == 0) * 1 / prevalence + (dflab[f"{i}_neut"] != 0) * 1 / (1 - prevalence)
            val = val / val.mean()
            assert (sum(val) - len(val)) < .01
            dflab.insert(loc=0, column=f'{i}_cw', value=val)
        cwdf = dflab[['sentence_id'] + [i + "_cw" for i in OUT_VARNAMES]]
        # merge the structured data onto the main df, and cut into training and test sets
        df = dflab.merge(str_all)
        assert df.shape[0] == dflab.shape[0]

        # get embeddings for fold
        embeddings = emb.reset_index(drop=True)
        assert df.shape[0] == embeddings.shape[0]
        # Convert text into matrix of tf-idf features:
        # id documents
        docs = dflab['sentence'].tolist()
        tfidf_ids = dflab['sentence_id'].tolist()
        # instantiate countvectorizer (turn off default stopwords)
        cv = CountVectorizer(analyzer='word', stop_words=None)
        # compute tf
        f_tf = cv.fit_transform(docs)
        # id additional stopwords: medlist_was_here_but_got_cut,
        # meds_was_here_but_got_cut, catv2_was_here_but_got_cut
        cuttext = '_was_here_but_got_cut'
        stopw = [i for i in list(cv.get_feature_names()) if re.search(cuttext, i)]
        # repeat countvec with full list of stopwords
        cv = CountVectorizer(analyzer='word', stop_words=stopw)
        # fit to data, then transform to count matrix
        f_tf = cv.fit_transform(docs)
        # fit to count matrix, then transform to tf-idf representation
        tfidf_transformer = TfidfTransformer()
        f_tfidf = tfidf_transformer.fit_transform(f_tf)
        # dimensionality reduction with truncated SVD
        svd_300 = TruncatedSVD(n_components=300, n_iter=5, random_state=9082020)
        svd_1000 = TruncatedSVD(n_components=1000, n_iter=5, random_state=9082020)
        # fit to  data & transform
        f_svd300 = pd.DataFrame(svd_300.fit_transform(f_tfidf))
        f_svd1000 = pd.DataFrame(svd_1000.fit_transform(f_tfidf))
        # add sentence ID to all of the TFIDF data frames
        f_svd300.insert(0, value=tfidf_ids, column='sentence_id')
        f_svd1000.insert(0, value=tfidf_ids, column='sentence_id')
        # sorting
        embeddings = embeddings.sort_values('sentence_id')
        f_svd300 = f_svd300.sort_values('sentence_id')
        f_svd1000 = f_svd1000.sort_values('sentence_id')
        # checking
        assert all(df.sentence_id.values == embeddings.sentence_id.values)
        assert all(df.sentence_id.values == f_svd300.sentence_id.values)
        assert all(df.sentence_id.values == f_svd1000.sentence_id.values)
        # sklearn stuff
        sklearn_dict['cv'] = cv
        sklearn_dict['tfidf_transformer'] = tfidf_transformer
        sklearn_dict['svd_300'] = svd_300
        sklearn_dict['svd_1000'] = svd_1000
        # Output for r
        df.to_csv(f"{self.ALdir}/processed_data/full_set/full_df.csv")
        embeddings.to_csv(f"{self.ALdir}/processed_data/full_set/full_embed_min_max_mean_SENT.csv")
        f_svd300.to_csv(f"{self.ALdir}/processed_data/full_set/full_svd300.csv")
        f_svd1000.to_csv(f"{self.ALdir}/processed_data/full_set/full_svd1000.csv")
        # case weights
        cwdf.to_csv(f"{self.ALdir}/processed_data/full_set/full_caseweights.csv")
        # sklearn artifacts
        write_pickle(sklearn_dict, f"{self.ALdir}/processed_data/full_set/full_sklearn_dict.pkl")
        print(f"Saved full data for training after CV")


def main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="batch string, i.e.: 00 or 01 or 02")
    p.add("--do_folds", action='store_true', help="process the data for all the folds")
    p.add("--do_full", action='store_true', help="process the full dataset for training")
    p.add("--do_final_80_20", action='store_true', help="create a 80/20 train-val split using the full training set")
    options = p.parse_args()
    batchstring = options.batchstring

    processor = DataProcessor(batchstring)
    if options.do_folds == True:
        for repeat in [1, 2, 3]:
            for fold in range(10):
                processor.process_fold(repeat=repeat, fold=fold)

    if options.do_full == True:
        processor.process_full_training_set()

    if options.do_final_80_20 == True:
        processor.process_fold(repeat=None, fold=None, final_80_20 = True)


if __name__ == '__main__':
    # self = DataProcessor('01')
    main()
