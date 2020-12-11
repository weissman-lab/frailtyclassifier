import copy
import os
import re
import sys
from time import process_time

import numpy as np
import pandas as pd
import tensorflow as tf
from gensim.models import KeyedVectors
from itertools import product
from keras.layers import Dense, Input, LSTM, Bidirectional, concatenate
from keras.models import Model
from keras.utils import to_categorical
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers.experimental.preprocessing import \
    TextVectorization

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000

def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def slidingWindow(sequence, winSize, step=1):
    # Verify the inputs
    # winSize is the total size of the window (including the center word)
    try:
        it = iter(sequence)
    except TypeError:
        raise Exception("**ERROR** sequence must be iterable.")
    if not ((type(winSize) == type(0)) and (type(step) == type(0))):
        raise Exception("**ERROR** type(winSize) and type(step) must be int.")
    if step > winSize:
        raise Exception("**ERROR** step must not be larger than winSize.")
    if winSize > len(sequence):
        raise Exception(
            "**ERROR** winSize must not be larger than sequence length.")
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - (winSize - 1)) / step)
    # Start half a window into the text
    # Create a window by adding words on either side of the center word
    for i in range(int(np.ceil((winSize - 1) / 2)) + 1,
                   (int(np.floor((winSize - 1) / 2)) + numOfChunks + 1) * step,
                   step):
        yield sequence[i - (int(np.ceil((winSize - 1) / 2)) + 1): i + int(
            np.floor((winSize - 1) / 2))]


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


# def kerasmodel(n_lstm, n_dense, n_units, dropout, l1_l2_pen):
#     nlp_input = Input(shape=(sentence_length,), name='nlp_input')
#     meta_input = Input(shape=(len(str_varnames),), name='meta_input')
#     x = cr_embed_layer(nlp_input)
#     for l in range(n_lstm - 1):
#         x = Bidirectional(LSTM(n_units, return_sequences=True,
#                                kernel_regularizer=l1_l2(l1_l2_pen)))(x)
#     y = Bidirectional(LSTM(n_units))(x)
#     for i in range(n_dense):
#         y = Dense(n_units, activation='relu',
#                   kernel_regularizer=l1_l2(l1_l2_pen))(y if i == 0 else drp)
#         drp = Dropout(dropout)(y)
#     concat = concatenate([drp, meta_input])
#     z = Dense(3, activation='sigmoid')(concat)
#     model = Model(inputs=[nlp_input, meta_input], outputs=[z])
#     return (model)

def acdkerasmodel(n_lstm, n_dense, n_units, dropout, l1_l2_pen):
    nlp_input = Input(shape=(sentence_length,), name='nlp_input')
    meta_input = Input(shape=(len(str_varnames),), name='meta_input')
    x = cr_embed_layer(nlp_input)
    for l in range(n_lstm):
        bid = Bidirectional(LSTM(n_units, return_sequences=True,
                               kernel_regularizer=l1_l2(l1_l2_pen)))(x)
    for i in range(n_dense):
        dense = Dense(n_units, kernel_regularizer=l1_l2(l1_l2_pen))(bid if i == 0 else drp)
        lru = LeakyReLU()(dense)
        drp = Dropout(dropout)(lru)
    flat = Flatten()(drp)
    concat = concatenate([flat, meta_input])
    z = Dense(3, activation='sigmoid')(concat)
    model = Model(inputs=[nlp_input, meta_input], outputs=[z])
    return (model)


# get experiment number from command line arguments
assert len(sys.argv) == 2, 'Exp number must be specified as an argument'
exp1 = sys.argv[1]
exp = f"exp{exp1}_nnet_str_WIN"
exp_SENT = f"exp{exp1}_nnet_str_SENT"

# get the correct directories
dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/",
        "/media/drv2/andrewcd2/frailty/output/", "/share/gwlab/frailty/"]
for d in dirs:
    if os.path.exists(d):
        datadir = d
if datadir == dirs[0]:  # mb
    outdir = f"{datadir}n_nets/{exp}/"
    outdir_SENT = f"{datadir}n_nets/{exp_SENT}/"
    pretr_embeddingsdir = f"{os.getcwd()}/embeddings/W2V_300_all/"
if datadir == dirs[1]:  # grace
    outdir = f"{os.getcwd()}/output/n_nets/{exp}/"
    outdir_SENT = f"{os.getcwd()}/output/n_nets/{exp_SENT}/"
    pretr_embeddingsdir = f"{os.getcwd()}/embeddings/W2V_300_all/"
if datadir == dirs[2]:  # azure
    outdir = f"{datadir}output/n_nets/{exp}/"
    outdir_SENT = f"{datadir}output/n_nets/{exp_SENT}/"
    pretr_embeddingsdir = "/share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/"
    datadir = f"{datadir}output/"

# makedir if missing
sheepish_mkdir(outdir)

# load the notes from 2018
notes_2018 = [i for i in os.listdir(datadir + "notes_labeled_embedded/") if
              int(i.split("_")[-2][1:]) < 13]

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
df = pd.concat([pd.read_csv(datadir + "notes_labeled_embedded/" + i) for i in
                notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)

# reset the index
df = df.reset_index()
# drop embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')].copy()

# set seed
seed = 111120

# define some useful constants
str_varnames = df2.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df2.columns if re.match("identity", i)]
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# make dummies for the outcomes
y_dums = pd.concat(
    [pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
df_dums = pd.concat([y_dums, df2['note']], axis=1)

# make "windows" of tokens to classify
count = 0
window3 = None
win_size = 11
for i in list(df2.note.unique()):
    # Create sliding window of 11 tokens for each note
    note_i = df2.loc[df2['note'] == i]
    chunks = slidingWindow(note_i['token'].tolist(), winSize=win_size)
    # now concatenate output from generator separated by blank space
    window = []
    for each in chunks:
        window.append(' '.join(each))
    # repeat the first and final windows (first and last few tokens
    # ((winSize-1)/2) will have off-center windows)
    window2 = list(np.repeat(window[0], np.floor((win_size - 1) / 2)))
    window2.extend(window)
    repeats_end = list(
        np.repeat(window2[len(window2) - 1], np.ceil((win_size - 1) / 2)))
    window2.extend(repeats_end)
    # repeat for all notes
    if window3 is None:
        window3 = window2
    else:
        window3.extend(window2)
    count += 1
# add windows to df
df2['window'] = window3

# label windows with heirarchical rule
# at each token, find the max for each dummy for an 11-token window
window_label = None
for i in list(df_dums.note.unique()):
    # important to copy the data in this step to avoid chained indexing
    note_i = df_dums.loc[df_dums['note'] == i].copy()
    for v in list(y_dums.columns):
        # rolling max
        note_i[f"any_{v}"] = note_i[f"{v}"].rolling(window=11, center=True,
                                                    min_periods=0).max()
    if window_label is None:
        window_label = note_i
    else:
        window_label = window_label.append(note_i, ignore_index=True)
#apply pos/neg/neutral label using heirarchical rule
#0 = neg, 1 = pos, 2 = neut (
for n in out_varnames:
    window_label[f"{n}_pos"] = np.where(
        (window_label[f"any_{n}_1"] == 1), 1, 0)
    window_label[f"{n}_neg"] = np.where(
        ((window_label[f"{n}_pos"] != 1) &
         (window_label[f"any_{n}_-1"] == 1)), 1,
        0)
    window_label[f"{n}_neut"] = np.where(
        ((window_label[f"{n}_pos"] != 1) &
         (window_label[f"{n}_neg"] != 1)), 1, 0)

# split into training and validation
# np.random.seed(seed)
# trnotes = np.random.choice(notes_2018_in_cndf,
#                            len(notes_2018_in_cndf) * 2 // 3, replace=False)
# tenotes = [i for i in notes_2018_in_cndf if i not in trnotes]
# trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
# tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

#get training/test split from _08_neural_nets_sentence.py
trnotes = list(pd.read_csv(f"{outdir_SENT}{exp_SENT}_train_notes.csv").iloc[:,1])
tenotes = list(df2[~df2.note.isin(trnotes)]['note'].unique())


# make categorical labels in correct tensor shape
tr_labels = []
te_labels = []
for n in out_varnames:
    r = window_label[window_label.note.isin(trnotes)][[f"{n}_neg", f"{n}_neut",
                                                       f"{n}_pos"]].to_numpy(dtype='float32').copy()
    e = window_label[window_label.note.isin(tenotes)][[f"{n}_neg", f"{n}_neut",
                                                       f"{n}_pos"]].to_numpy(dtype='float32').copy()
    tr_labels.append(r)
    te_labels.append(e)

# caseweights - weight non-neutral tokens by the inverse of their prevalence
tr_cw = []
for v in out_varnames:
    non_neutral = np.array(np.sum(
        y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]],
        axis=1)).astype('float32')
    nnweight = 1 / np.mean(non_neutral[df_dums.note.isin(trnotes)])
    caseweights = np.ones(df_dums.shape[0])
    caseweights[non_neutral.astype(bool)] *= nnweight
    tr_caseweights = caseweights[df_dums.note.isin(trnotes)]
    tr_cw.append(tr_caseweights)

# structured data tensors
# first, scale the structured data
scaler = StandardScaler()
scaler.fit(df2[str_varnames].loc[df2.note.isin(trnotes)])
sdf = copy.deepcopy(df2)
sdf[str_varnames] = scaler.transform(df2[str_varnames])
# get training & test data in numpy arrays
train_struc = np.asarray(sdf[sdf.note.isin(trnotes)][str_varnames]).astype(
    'float32')
test_struc = np.asarray(sdf[sdf.note.isin(tenotes)][str_varnames]).astype(
    'float32')

# create vocabulary index
train_windows = df2[df2.note.isin(trnotes)]['window']
test_windows = df2[df2.note.isin(tenotes)]['window']
# vectorize text (must use venv_ft environment -- not a conda environment,
# which only allows tensorflow 2.0 on mac)
vectorizer = TextVectorization(max_tokens=None, #unlimited vocabulary size
                               output_sequence_length=win_size,
                               standardize=None)  # this is CRITICAL -- default
# will strip '_' and smash multi-word-expressions together
vectorizer.adapt(np.array(train_windows))
# now each window is represented by a vector that maps each word with an integer

# vectorize training & test windows
x_train = vectorizer(np.array([[s] for s in train_windows])).numpy()
x_test = vectorizer(np.array([[s] for s in test_windows])).numpy()

# get the vocabulary from our vectorized text
vocab = vectorizer.get_vocabulary()
# make a dictionary mapping words to their indices
word_index = dict(zip(vocab, range(len(vocab))))

# Load word2vec model
cr_embed = KeyedVectors.load(f"{pretr_embeddingsdir}w2v_oa_all_300d.bin",
                             mmap='r')
# create an embedding matrix (embeddings mapped to word indices)
# adding 1 ensures that there is a row of zeros in the embedding matrix for
# words in the text that are not in the embeddings vocab
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

# test for lack of obs in test or training data
test_zero_obs(tr_labels)
test_zero_obs(te_labels)


#make hyperparameter grid
# hp_grid = {'n_lstm': [1, 3],
#            'n_dense': [1, 3],
#            'n_units': [64, 512],
#            'sample_weights': [False, True],
#            'dropout': np.linspace(0.01, 0.5, 2),
#            'l1_l2': np.linspace(1e-8, 1e-4, 2)}
hp_grid = {'n_lstm': [1],
           'n_dense': [2],
           'n_units': [256],
           'sample_weights': [True],
           'dropout': [0.15250199],
           'l1_l2': [0.0000000137]}
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
#iterate over hp_grid
for r in range(hp_grid.shape[0]):
    # iterate over the frailty aspects
    for m in range(len(tr_labels)):
        protime_start = process_time()
        frail_lab = out_varnames[m]
        # model name
        mod_name = f"bl{hp_grid.iloc[r].n_lstm}_den{hp_grid.iloc[r].n_dense}_u{hp_grid.iloc[r].n_units}_sw"
        fr_mod = f"{frail_lab}_{mod_name}"
        model_2 = acdkerasmodel(hp_grid.iloc[r].n_lstm,
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
    #output all process times
    all_protimes_out = pd.concat(all_protimes, ignore_index=True)
    all_protimes_out = all_protimes_out.rename(index=index_names)
    all_protimes_out.to_csv(f"{outdir}{exp}_{mod_name}_protime.csv")