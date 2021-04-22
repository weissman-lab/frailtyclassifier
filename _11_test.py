import os
import re
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
from utils.constants import SENTENCE_LENGTH, OUT_VARNAMES, STR_VARNAMES, TAGS
from utils.misc import read_pickle, test_nan_inf, sheepish_mkdir, send_message_to_slack
from utils.prefit import make_model

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


class TestPredictor:
    def __init__(self, batchstring, task, use_training_dict = False, save = True):
        assert task in ['multi', 'Resp_imp', 'Msk_prob', 'Nutrition', 'Fall_risk']
        self.outdir = f"./output/"
        self.datadir = f"./data/"
        self.ALdir = f"{self.outdir}saved_models/AL{batchstring}/"
        self.batchstring = batchstring
        self.task = task
        self.save = save
        self.use_training_dict = use_training_dict
        self.suffix = "" if self.task == "multi" else f"_{self.task}"
        # things defined in methods
        self.df = None
        self.strdat = None
        self.training_data = None
        self.str_varnames = None
        self.model = None
        self.vectorizer = None

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
        df = df.drop(columns=['sent_start', 'length'] + [i for i in df.columns if 'identity_' in i])
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
        self.df = df_label

    def compile_structured_data(self):
        self.training_data = pd.read_csv(f"{self.ALdir}processed_data/full_set/full_df.csv", index_col=0)
        self.str_varnames = [i for i in self.training_data.columns if re.match("pca[0-9]", i)]
        skd = read_pickle(f"{self.ALdir}processed_data/full_set/full_sklearn_dict.pkl")
        strdat = pd.read_csv(f"{self.outdir}structured_data_merged_cleaned.csv",
                             index_col=0)
        strdat = strdat.drop(columns=['RACE', 'LANGUAGE', 'MV_RACE', 'MV_LANGUAGE'])
        tm = copy.deepcopy(self.df[['PAT_ID', 'sentence_id']])
        tm['month'] = tm.sentence_id.apply(lambda x: int(x.split("_")[2][1:]))
        strdat = strdat.merge(tm[['PAT_ID', 'month']].drop_duplicates())
        strdat_imp = skd['imputer'].transform(strdat[STR_VARNAMES])
        scaled = skd['scaler_in'].transform(strdat_imp)
        rot = skd['pca'].transform(scaled)
        rot_scaled = skd['scaler_out'].transform(rot)
        str_all = pd.concat([strdat[['PAT_ID', 'month']], pd.DataFrame(rot_scaled)], axis=1)
        str_all.columns = ['PAT_ID', 'month'] + ['pca' + str(i) for i in range(rot_scaled.shape[1])]
        #
        self.strdat = str_all

    def reconstitute_model(self):
        mod_dict = read_pickle(f"{self.ALdir}final_model/model_final_{self.batchstring}{self.suffix}.pkl")

        if self.use_training_dict == True:
            sents = self.training_data.sentence
        else:
            sents = self.df.sentence

        model, vectorizer = make_model(emb_path=f"{self.datadir}w2v_oa_all_300d.bin",
                                       sentence_length=SENTENCE_LENGTH,
                                       meta_shape=len(self.str_varnames),
                                       tags=OUT_VARNAMES if self.task == 'multi' else [self.task],
                                       train_sent=sents,
                                       l1_l2_pen=mod_dict['config']['l1_l2_pen'],
                                       n_units=mod_dict['config']['n_units'],
                                       n_dense=mod_dict['config']['n_dense'],
                                       dropout=mod_dict['config']['dropout'])
        weights = model.get_weights()
        for i in range(1, len(weights)): # ignore the first weight matrix, which is the embeddings
            weights[i] = mod_dict['weights'][i]
        model.set_weights(weights)
        # model.set_weights(mod_dict['weights'])
        self.model = model
        self.vectorizer = vectorizer

    def predict(self):
        self.df['month'] = self.df.sentence_id.apply(lambda x: int(x.split("_")[2][1:]))
        pred_df = self.df.merge(self.strdat)
        text = self.vectorizer(np.array([[s] for s in pred_df.sentence]))
        labels = []
        if self.task == 'multi':
            for n in TAGS:
                lab = tf.convert_to_tensor(pred_df[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]], dtype='float32')
                labels.append(lab)
            assert all([all(tf.reduce_mean(tf.cast(i, dtype='float32'), axis=0) % 1 > 0) for i in labels])
        else:
            labels = tf.convert_to_tensor(pred_df[[f"{self.task}_neg",
                                                   f"{self.task}_neut",
                                                   f"{self.task}_pos"]], dtype='float32')
            assert all(tf.reduce_mean(tf.cast(labels, dtype='float32'), axis=0) % 1 > 0)
        struc = tf.convert_to_tensor(pred_df[self.str_varnames], dtype='float32')
        test_nan_inf(text)
        test_nan_inf(labels)
        test_nan_inf(struc)
        yhat = self.model.predict([text, struc])
        if self.task == 'multi':
            yhat_df = []
            for i, yh in enumerate(yhat):
                yhat_df.append(
                    pd.DataFrame(yh, columns=[f"{TAGS[i]}_neg_pred", f"{TAGS[i]}_neut_pred", f"{TAGS[i]}_pos_pred"]))
            out = pd.concat([pred_df, pd.concat(yhat_df, axis=1)], axis=1)
            out = out.drop(columns=[i for i in out.columns if "pca" in i])
        else:
            yhat_df = pd.DataFrame(yhat,
                                   columns=[f"{self.task}_neg_pred", f"{self.task}_neut_pred", f"{self.task}_pos_pred"])
            out = pd.concat([pred_df, yhat_df], axis=1)
            for k in ['pca'] + [j for j in TAGS if self.task != j]:
                out = out.drop(columns=[i for i in out.columns if k in i])
        return out

    def run(self):
        self.compile_text()
        self.compile_structured_data()
        self.reconstitute_model()
        preds = self.predict()
        sheepish_mkdir(f"{self.ALdir}final_model/test_preds")
        if self.save == True:
            preds.to_csv(f"{self.ALdir}final_model/test_preds/test_preds_AL{self.batchstring}{self.suffix}.csv")




def main():
    for tag in [TAGS + 'multi']:
        for bs in ["0" + str(i + 1) for i in range(5)]:
            try:
                TestPredictor(batchstring=bs, task=tag).run()
            except:
                send_message_to_slack(f"problem with batch {bs} tag {tag}")




if __name__ == "__main__":
    pass
    main()
    # TestPredictor(batchstring='03', task='Msk_prob').run()
    # TestPredictor(batchstring='03', task='Fall_risk').run()
    # TestPredictor(batchstring='03', task='multi').run()

# self = TestPredictor(batchstring='03', task='multi', use_training_dict = False, save = False)
#
# xx = preds
# # xx = pd.read_csv('/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/saved_models/AL03/final_model/test_preds/preds_Fall_risk.csv')
#
# y = xx[[i for i in xx.columns if any([j in i for j in TAGS]) and 'pred' not in i]]
# yhat = xx[[i for i in xx.columns if any([j in i for j in TAGS]) and 'pred' in i]]
# yhat = yhat[[i+"_pred" for i in y.columns]]
#
# mse = ((yhat.values - y.values)**2).mean()
# mst = ((y.values.mean(axis = 0) - y.values)**2).mean()
# 1-mse/mst
#
# import matplotlib.pyplot as plt
# plt.hist(yhat.Fall_risk_neg_pred)
# plt.show()