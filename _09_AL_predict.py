import copyfrom configargparse import ArgParserimport reimport osimport pandas as pdimport tensorflow as tffrom utils.prefit import make_modelfrom utils.misc import (read_pickle, sheepish_mkdir, inv_logit,                        send_message_to_slack, entropy)from utils.constants import SENTENCE_LENGTH, TAGS, STR_VARNAMESimport spacyfrom _05_ingest_to_sentence import set_custom_boundariesimport numpy as npimport timebatchstring = '03'outdir = f"{os.getcwd()}/output/"datadir = f"{os.getcwd()}/data/"ALdir = f"{outdir}saved_models/AL{batchstring}/"class BatchPredictor():    def __init__(self,digits):            #################        # load data and subset        self.df = pd.read_csv(f"{ALdir}processed_data/full_set/full_df.csv", index_col = 0)            allnotes = pd.read_pickle(f"{outdir}conc_notes_df.pkl")        allnotes = allnotes.loc[(allnotes.LATEST_TIME < "2019-01-01") &                                (~allnotes.PAT_ID.isin(self.df.PAT_ID))]        allnotes = allnotes.loc[allnotes.PAT_ID.apply(lambda x: x[-2:]) == digits]        allnotes.insert(0, "uid", allnotes.PAT_ID + "_" + allnotes.LATEST_TIME.dt.month.astype(str))        strdat = pd.read_csv(f"{outdir}structured_data_merged_cleaned.csv", index_col = 0)           strdat.insert(0, "uid", strdat.PAT_ID + "_" + strdat.month.astype(str))        self.allnotes = allnotes.loc[allnotes.uid.isin(strdat.uid)]        self.strdat = strdat.loc[strdat.uid.isin(self.allnotes.uid)]        self.str_varnames = [i for i in self.df.columns if re.match("pca[0-9]",i)]        self.scalers = read_pickle(f"{ALdir}/processed_data/full_set/full_sklearn_dict.pkl")        ####################        # spacy        self.sci_nlp = spacy.load("en_core_sci_md", disable=['tagger', 'ner'])        self.sci_nlp.add_pipe(set_custom_boundaries, before="parser",                              name='set_custom_boundaries')            #######################        # Load the model and vectorizer        mod_dict = read_pickle(f"{ALdir}/final_model/model_final_{batchstring}.pkl")        self.model, self.vectorizer = make_model(emb_path = f"{datadir}w2v_oa_all_300d.bin",                                                 sentence_length = SENTENCE_LENGTH,                                                 meta_shape = len(self.str_varnames),                                                 tags = TAGS,                                                 train_sent = self.df['sentence'],                                                 l1_l2_pen = mod_dict['config']['l1_l2_pen'],                                                 n_units = mod_dict['config']['n_units'],                                                 n_dense = mod_dict['config']['n_dense'],                                                 dropout = mod_dict['config']['dropout'])        self.model.compile(loss='categorical_crossentropy',                      optimizer=tf.keras.optimizers.Adam(1e-4))        self.model.set_weights(mod_dict['weights'])    def sentencize(self, note):        res = self.sci_nlp(note)        sentences = []        s = []        for i, tok in enumerate(res):            if (bool(tok.is_sent_start)) & (len(s) > 0):                newsent = " ".join(s)                sentences.append(newsent)                s = []            s.append(str(tok))        sentences.append(" ".join(s))        text = self.vectorizer(np.array([[s] for s in sentences]))        return text    def process_structured(self, row, nsent):        str_mat = pd.concat([row for i in range(nsent)])[STR_VARNAMES]        str_mat = self.scalers['imputer'].transform(str_mat)        str_mat = self.scalers['scaler_in'].transform(str_mat)        str_mat = self.scalers['pca'].transform(str_mat)        str_mat = self.scalers['scaler_out'].transform(str_mat)        struc = tf.convert_to_tensor(str_mat, dtype = 'float32')        return struc    def predict(self, ii):        text = self.sentencize(self.allnotes.combined_notes.iloc[ii])        struc = self.process_structured(self.strdat.loc[self.strdat.uid == self.allnotes.uid.iloc[ii]],                                         nsent = text.shape[0])        pred = self.model.predict([text, struc])        return pred            def get_H_stats(self, pred):        '''pred is a list of predictions for each aspect'''        nsent = pred[0].shape[0]        hmat = np.stack([entropy(i) for i in pred])        average_entropy = np.nanmean(hmat)        max_entropy = np.nanmax(hmat)        hflat = hmat.flatten()        average_entropy_above_median = np.mean(hflat[hflat>np.nanmedian(hflat)])        hcut = copy.deepcopy(hmat)        for i in range(hcut.shape[0]):            hcut[i,:][hcut[i,:]<np.nanmedian(hcut[i,:])] = np.nan        hcut = np.max(hcut, axis = 0)        average_entropy_above_median_notewise = np.nanmean(hcut)        return dict(nsent = nsent,                    average_entropy=average_entropy,                    max_entropy=max_entropy,                    average_entropy_above_median=average_entropy_above_median,                    average_entropy_above_median_notewise=average_entropy_above_median_notewise)    def loop_predict(self):        outlist = []        for ii in range(len(self.allnotes)):            pred = self.predict(ii)            out = self.get_H_stats(pred)            out['uid'] = self.allnotes.uid.iloc[ii]            print(out)            outlist.append(out)        dfout = pd.DataFrame(outlist)    predictor = BatchPredictor('01')dfout = predictor.loop_predict()def main():    p = ArgParser()    p.add("-b", "--batchstring", help="the batch number", type=str)    options = p.parse_args()    batchstring = options.batchstring        outdir = f"{os.getcwd()}/output/"    ALdir = f"{outdir}saved_models/AL{batchstring}/"    sheepish_mkdir(f"{ALdir}final_model/preds")    digit_vec = np.random.choice([str(i)[1:] for i in range(100,200)])    for digit in digit_vec:        try:            predictor = BatchPredictor(digit)        except:            ip = os.popen("/sbin/ip -o -4 addr list eth0 | awk '{print $4}' | cut -d/ -f1").read()            send_message_to_slack(f"problem at {ip}")if __name__ == "__main__":    main()