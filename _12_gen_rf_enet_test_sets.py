
import os
import re
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.constants import SENTENCE_LENGTH, OUT_VARNAMES, STR_VARNAMES, TAGS
from utils.misc import read_pickle, test_nan_inf, sheepish_mkdir, send_message_to_slack
from _11_test import TestPredictor

class RModelDataMaker(TestPredictor):
    def __init__(self, batchstring):
        super().__init__(batchstring, task = 'multi')
        self.emb = None

    def compile_text(self):
        '''
        Loop through test notes and put them in the form required to feed the model
        :return:
        '''
        testnotes = os.listdir(f"{self.outdir}notes_labeled_embedded_SENTENCES/test")
        testnotes = [i for i in testnotes if int(i.split("_")[3][1:]) > 11]
        df = pd.concat([pd.read_csv(f"{self.outdir}notes_labeled_embedded_SENTENCES/test/{i}",
                                    index_col=0,
                                    dtype=dict(PAT_ID=str)) for i in testnotes])
        df = df.drop(columns=['sent_start', 'length'])
        sent_str = df.sentence.apply(lambda x: "".join(["0" for i in range(6 - len(str(x)))]) + str(x))
        df.insert(2, 'sentence_id', df.note + "_sent" + sent_str)
        df.rename(columns={'sentence': 'sentence_in_note'}, inplace=True)
        y_dums = pd.concat(
            [pd.get_dummies(df[[i]].astype(str)) for i in OUT_VARNAMES], axis=1)
        df = pd.concat([y_dums, df], axis=1)
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
        df_label = df_label.reset_index(drop=False)
        for n in OUT_VARNAMES:
            df_label[f"{n}_neg"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"any_{n}_neg"] == 1)), 1,
                0)
            df_label[f"{n}_neut"] = np.where(
                ((df_label[f"{n}_pos"] != 1) & (df_label[f"{n}_neg"] != 1)), 1, 0)
        # drop extra columns
        df_label = df_label.loc[:, ~df_label.columns.str.startswith('any_')].copy()
        df_label.insert(loc=0, column='PAT_ID', value=df_label.sentence_id.apply(lambda x: x.split("_")[-2]))
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
        assert all(df_label.sentence_id.isin(emb.sentence_id))
        assert all(emb.sentence_id.isin(df_label.sentence_id))
        self.df = df_label
        self.emb = emb

    def generate_files(self):
        df = self.df.merge(self.strdat)
        assert df.shape[0] == self.df.shape[0]

self.df.loc[self.df.PAT_ID.isin(self.strdat.PAT_ID)]
self.df.shape
df.shape
self.strdat.drop_duplicates().shape
self.strdat.PAT_ID.nunique()

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



    def run(self):
        self.compile_text()
        self.compile_structured_data()


if __name__ == "__main__":
    self = RModelDataMaker(batchstring = '03')
