import os
import sys
from itertools import product
from time import process_time

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from keras.layers import Dense, Input, LSTM, Bidirectional, concatenate, LeakyReLU, Dropout, Flatten
from keras.models import Model
from sklearn.metrics import brier_score_loss
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization
from tensorflow.keras.regularizers import l1_l2

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def scaled_brier(obs, pred):
    numerator = brier_score_loss(obs, pred)
    denominator = brier_score_loss(obs, [np.mean(obs)] * len(obs))
    return (1 - (numerator / denominator))


# test for missing values before training
def test_nan_inf(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nan.')
    if np.isinf(tensor).any():
        raise ValueError('Tensor contains inf.')


# test for lack of obs in test or training data
def test_zero_obs(tensor):
    for l in range(len(tensor)):
        for c in range(tensor[l].shape[1]):
            if sum(tensor[l][:, c]) == 0:
                raise ValueError('No observations in test set.')


def expand_grid(grid):
    return pd.DataFrame([row for row in product(*grid.values())],
                        columns=grid.keys())

def kerasmodel(n_lstm, n_dense, n_units, dropout, l1_l2_pen):
    nlp_input = Input(shape=(sentence_length,), name='nlp_input')
    meta_input = Input(shape=(len(str_varnames),), name='meta_input')
    x = cr_embed_layer(nlp_input)
    for l in range(n_lstm - 1):
        x = Bidirectional(LSTM(n_units, return_sequences=True,
                               kernel_regularizer=l1_l2(l1_l2_pen)))(x)
    y = Bidirectional(LSTM(n_units))(x)
    for i in range(n_dense):
        y = Dense(n_units, activation='relu',
                  kernel_regularizer=l1_l2(l1_l2_pen))(y if i == 0 else drp)
        drp = Dropout(dropout)(y)
    concat = concatenate([drp, meta_input])
    z = Dense(3, activation='sigmoid')(concat)
    model = Model(inputs=[nlp_input, meta_input], outputs=[z])
    return (model)


# get experiment number from command line arguments
assert len(sys.argv) == 2, 'Exp number must be specified as an argument'
exp = sys.argv[1]
exp = f"exp{exp}"

# get the correct directories
dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/",
        "/media/drv2/andrewcd2/frailty/", "/share/gwlab/frailty/"]
for d in dirs:
    if os.path.exists(d):
        rootdir = d
if rootdir == dirs[0]:  # mb
    trtedatadir = f"{rootdir}output/notes_preprocessed_SENTENCES/trtedata/"
    pretr_embeddingsdir = f"{rootdir}/embeddings/W2V_300_all/"
    outdir = f"{rootdir}output/n_nets/{exp}/"
if rootdir == dirs[1]:  # grace
    trtedatadir = f"{os.getcwd()}/output/notes_preprocessed_SENTENCES/trtedata/"
    pretr_embeddingsdir = f"{os.getcwd()}/embeddings/W2V_300_all/"
    outdir = f"{os.getcwd()}/output/n_nets/{exp}/"
if rootdir == dirs[2]:  # azure
    trtedatadir = f"{rootdir}output/notes_preprocessed_SENTENCES/trtedata/"
    pretr_embeddingsdir = "/share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/"
    outdir = f"{rootdir}output/n_nets/{exp}/"

# makedir if missing
sheepish_mkdir(outdir)

# define constants
out_varnames = ['Msk_prob', 'Nutrition', 'Resp_imp', 'Fall_risk']
repeats = range(3)
folds = range(10)

# load, prep, and run models by fold
for r in repeats:
    for f in folds:
        f_tr = pd.read_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_tr_df.csv")
        f_te = pd.read_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_te_df.csv")
        f_tr_cw = pd.read_csv(f"{trtedatadir}r{r + 1}_f{f + 1}_tr_cw.csv")[
            list(s + '_cw' for s in out_varnames)]

        # make categorical labels in correct tensor shape
        tr_labels = []
        te_labels = []
        for n in out_varnames:
            tr = f_tr[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]].to_numpy(
                dtype='float32').copy()
            te = f_te[[f"{n}_neg", f"{n}_neut", f"{n}_pos"]].to_numpy(
                dtype='float32').copy()
            tr_labels.append(tr)
            te_labels.append(te)

        # structured data tensors
        pca_cols = [c for c in f_tr.columns if 'pc_' in c]
        train_struc = np.asarray(f_tr[pca_cols]).astype('float32')
        test_struc = np.asarray(f_te[pca_cols]).astype('float32')

        # set standard sentence length. Inputs will be truncated or padded
        # if necessary
        sentence_length = 18
        # create vocabulary index
        train_sent = f_tr['sentence']
        test_sent = f_te['sentence']
        # vectorize text (FYI on mac, must use venv_ft environment (not a conda
        # environment), to access tensorflow 2.0
        vectorizer = TextVectorization(max_tokens=None,  # unlimited vocab size
                                       output_sequence_length=sentence_length,
                                       standardize=None)  # this is CRITICAL --
        # default will strip '_' and smash multi-word-expressions together
        vectorizer.adapt(np.array(train_sent))
        # now each window is represented by a vector that maps each word with
        # an integer

        # vectorize training & test windows
        x_train = vectorizer(np.array([[s] for s in train_sent])).numpy()
        x_test = vectorizer(np.array([[s] for s in test_sent])).numpy()

        # get the vocabulary from our vectorized text
        vocab = vectorizer.get_vocabulary()
        # make a dictionary mapping words to their indices
        word_index = dict(zip(vocab, range(len(vocab))))

        # Load word2vec model
        cr_embed = KeyedVectors.load(f"{pretr_embeddingsdir}w2v_oa_all_300d.bin",
                                     mmap='r')
        # create an embedding matrix (embeddings mapped to word indices)
        # adding 1 ensures that there is a row of zeros in the embedding matrix
        # for words in the text that are not in the embeddings vocab
        num_words = len(word_index) + 1
        embedding_len = 300
        embedding_matrix = np.zeros((num_words, embedding_len))
        # add embeddings for each word to the matrix
        hits = 0
        misses = 0
        for word, i in word_index.items():
            # by default, words not found in embedding index will be all-zeros
            if word in cr_embed.wv.vocab:
                embedding_matrix[i] = cr_embed.wv.word_vec(word)
                hits += 1
            else:
                misses += 1
        print("Converted %s words (%s misses)" % (hits, misses))

        # load embeddings matrix into an Embeddings layer
        cr_embed_layer = Embedding(embedding_matrix.shape[0],
                                   embedding_matrix.shape[1],
                                   embeddings_initializer=keras.initializers.Constant(
                                       embedding_matrix),
                                   trainable=False)

        # test for missing values before training
        test_nan_inf(x_train)
        test_nan_inf(x_test)
        test_nan_inf(tr_labels)
        test_nan_inf(te_labels)
        test_nan_inf(train_struc)
        test_nan_inf(test_struc)
        # test for missing obs in test or training data
        test_zero_obs(tr_labels)
        test_zero_obs(te_labels)

        # hyperparameter grid
        hp_grid = {'n_lstm': [1, 3],
                   'n_dense': [1, 3],
                   'n_units': [64, 512],
                   'sample_weights': [False, True],
                   'dropout': np.linspace(0.01, 0.5, 2),
                   'l1_l2': np.linspace(1e-8, 1e-4, 2)}
        hp_grid = expand_grid(hp_grid)

        # set parameters for all models
        best_batch_s = 32
        epochs = 1000
        tr_loss_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                 patience=20,
                                                                 restore_best_weights=True)
        # set lists for output
        deep_loss = []
        deep_val_loss = []
        model_name = []
        train_sbriers = []
        test_sbriers = []
        all_protimes = []
        # iterate over hp_grid
        for r in range(hp_grid.shape[0]):
            # iterate over the frailty aspects
            for m in range(len(tr_labels)):
                protime_start = process_time()
                frail_lab = out_varnames[m]
                # model name
                mod_name = f"bl{hp_grid.iloc[r].n_lstm}_den{hp_grid.iloc[r].n_dense}_u{hp_grid.iloc[r].n_units}_sw"
                fr_mod = f"{frail_lab}_{mod_name}"
                model_2 = kerasmodel(hp_grid.iloc[r].n_lstm,
                                        hp_grid.iloc[r].n_dense,
                                        hp_grid.iloc[r].n_units,
                                        hp_grid.iloc[r].dropout,
                                        hp_grid.iloc[r].l1_l2_pen)
                model_2.compile(loss='categorical_crossentropy',
                                optimizer=tf.keras.optimizers.Adam(1e-4),
                                metrics=['acc'])
                # fit model
                history = model_2.fit([x_train, train_struc],
                                      tr_labels[m],
                                      validation_data=(
                                          [x_test, test_struc], te_labels[m]),
                                      epochs=epochs,
                                      batch_size=best_batch_s,
                                      sample_weight=tr_cw[m] if hp_grid.iloc[r].sample_weights is True else None,
                                      callbacks=[tr_loss_earlystopping])
                # add loss to list
                deep_loss.append(history.history['loss'])
                deep_val_loss.append(history.history['val_loss'])
                model_name.append(fr_mod)
                # save as df
                tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
                val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
                index_names = dict({1: fr_mod})
                col_names = dict(
                    zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
                tr_m_loss = tr_m_loss.rename(index=index_names, columns=col_names)
                val_m_loss = val_m_loss.rename(index=index_names, columns=col_names)
                tr_m_loss.to_csv(f"{outdir}{fr_mod}_train_loss.csv")
                val_m_loss.to_csv(f"{outdir}{fr_mod}_val_loss.csv")
                # make predictions on training data
                tr_probs = model_2.predict([x_train, train_struc])
                # scaled brier for each class
                tr_sb = []
                for i in range(3):
                    tr_sb.append(scaled_brier(tr_labels[m][:, i], tr_probs[:, i]))
                tr_sb = pd.DataFrame(tr_sb).transpose().rename(
                    columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
                train_sbriers.append(tr_sb)
                # save sbrier
                tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
                # make predictions on testing data
                te_probs = model_2.predict([x_test, test_struc])
                # save predictions
                pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
                # scaled briers for each class
                te_sb = []
                for i in range(3):
                    te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
                te_sb = pd.DataFrame(te_sb).transpose().rename(
                    columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
                test_sbriers.append(te_sb)
                # save sbrier
                te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")
                # save process time
                protime_end = process_time()
                protime = {'start': protime_start, 'end': protime_end,
                           'duration': (protime_end - protime_start)}
                protime_duration = pd.DataFrame([protime], columns=protime.keys())
                protime_duration.to_csv(f"{outdir}{fr_mod}_protime.csv")
                all_protimes.append(protime_duration)

            # early stopping causes differences in epochs -- pad with NA so columns match
            train_loss = np.ones(
                (len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
            val_loss = train_loss.copy()
            for i, c in enumerate(deep_loss):
                train_loss[i, :len(c)] = c
            train_loss = pd.DataFrame(train_loss)
            for i, c in enumerate(deep_val_loss):
                val_loss[i, :len(c)] = c
            val_loss = pd.DataFrame(val_loss)
            # rename index and columns
            index_names = dict(zip((range(len(model_name))), model_name))
            col_names = dict(
                zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
            train_loss = train_loss.rename(index=index_names, columns=col_names)
            val_loss = val_loss.rename(index=index_names, columns=col_names)
            # save
            train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
            val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
            # combine all sbriers together and save
            train_sbrier_out = pd.concat(train_sbriers, ignore_index=True)
            test_sbrier_out = pd.concat(test_sbriers, ignore_index=True)
            train_sbrier_out = train_sbrier_out.rename(index=index_names)
            test_sbrier_out = test_sbrier_out.rename(index=index_names)
            train_sbrier_out.to_csv(f"{outdir}{exp}_{mod_name}_train_sbrier.csv")
            test_sbrier_out.to_csv(f"{outdir}{exp}_{mod_name}_test_sbrier.csv")
            # output all process times
            all_protimes_out = pd.concat(all_protimes, ignore_index=True)
            all_protimes_out = all_protimes_out.rename(index=index_names)
            all_protimes_out.to_csv(f"{outdir}{exp}_{mod_name}_protime.csv")
