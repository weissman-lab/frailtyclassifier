import os
import sys
import re
import pandas as pd
import numpy as np
from itertools import product
import copy
from sklearn.preprocessing import StandardScaler
from timeit import default_timer as timer
from gensim.models import KeyedVectors
from _99_project_module import inv_logit
import tensorflow as tf
from tensorflow import keras
from keras.utils import to_categorical
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Embedding
from keras.models import Model
from keras.layers import Dense, Input, LSTM, Bidirectional, concatenate
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
        raise Exception("**ERROR** winSize must not be larger than sequence length.")
    # Pre-compute number of chunks to emit
    numOfChunks = int((len(sequence) - (winSize-1)) / step)
    # Start half a window into the text
    # Create a window by adding words on either side of the center word
    for i in range(int(np.ceil((winSize-1)/2))+1, (int(np.floor((winSize-1)/2))+numOfChunks+1) * step, step):
        yield sequence[i - (int(np.ceil((winSize-1)/2))+1) : i + int(np.floor((winSize-1)/2))]

def brier_score(obs, pred):
    return np.mean((obs - pred) ** 2)

def scaled_brier(obs, pred):
    numerator = brier_score(obs,pred)
    denominator = brier_score(obs, np.mean(obs))
    return(1 - (numerator/denominator))

#get experiment number from command line arguments
assert len(sys.argv) == 2, 'Exp number must be specified as an argument'
exp = sys.argv[1]
exp = f"exp{exp}_nnet_str"

#get the correct directories
dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/", "/media/drv2/andrewcd2/frailty/output/", "/share/gwlab/frailty/"]
for d in dirs:
    if os.path.exists(d):
        datadir = d
if datadir == dirs[0]: #mb
    outdir = f"{datadir}n_nets/{exp}/"
    pretr_embeddingsdir = f"{os.getcwd()}/embeddings/W2V_300_all/"
if datadir == dirs[1]: #grace
    outdir = f"{os.getcwd()}/output/n_nets/{exp}/"
    pretr_embeddingsdir = f"{os.getcwd()}/embeddings/W2V_300_all/"
if datadir == dirs[2]: #azure
    outdir = f"{datadir}output/n_nets/{exp}/"
    pretr_embeddingsdir = "/share/acd-azure/pwe/output/built_models/OA_ALL/W2V_300/"
    datadir = f"{datadir}output/"

#makedir if missing
sheepish_mkdir(outdir)

# load the notes from 2018
notes_2018 = [i for i in os.listdir(datadir + "notes_labeled_embedded/") if int(i.split("_")[-2][1:]) < 13]

# drop the notes that aren't in the concatenated notes data frame
# some notes got labeled and embedded but were later removed from the pipeline
# on July 14 2020, due to the inclusion of the 12-month ICD lookback
cndf = pd.read_pickle(f"{datadir}conc_notes_df.pkl")
cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
cndf['month'] = cndf.LATEST_TIME.dt.month + (
    cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
uidstr = ("m"+cndf.month.astype(str)+"_"+cndf.PAT_ID+".csv").tolist()

notes_2018_in_cndf = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) in uidstr]
notes_excluded = [i for i in notes_2018 if "_".join(i.split("_")[-2:]) not in uidstr]
assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)

df = pd.concat([pd.read_csv(datadir + "notes_labeled_embedded/" + i) for i in notes_2018_in_cndf])
df.drop(columns='Unnamed: 0', inplace=True)

# reset the index
df = df.reset_index()
#make smaller practice set - take first 100 tokens for each note
#df2 = df.groupby('note').head(100).copy()
#df2 = df2.reset_index()
# drop embeddings
df2 = df.loc[:, ~df.columns.str.startswith('identity')].copy()

seed = 111120

# define some useful constants
str_varnames = df2.loc[:, "n_encs":'MV_LANGUAGE'].columns.tolist()
embedding_colnames = [i for i in df2.columns if re.match("identity", i)]
out_varnames = df2.loc[:, "Msk_prob":'Fall_risk'].columns.tolist()
input_dims = len(embedding_colnames) + len(str_varnames)

# make "windows" of tokens to classify
count = 0
window3 = None
win_size = 11
for i in list(df2.note.unique()):
    #Create sliding window of 11 tokens for each note
    note_i = df2.loc[df2['note'] == i]
    chunks = slidingWindow(note_i['token'].tolist(), winSize=win_size)
    #now concatenate output from generator separated by blank space
    window = []
    for each in chunks:
        window.append(' '.join(each))
    #repeat the first and final windows (first and last few tokens ((winSize-1)/2) will have off-center windows)
    window2 = list(np.repeat(window[0], np.floor((win_size-1)/2)))
    window2.extend(window)
    repeats_end = list(np.repeat(window2[len(window2)-1], np.ceil((win_size-1)/2)))
    window2.extend(repeats_end)
    #repeat for all notes
    if window3 is None:
        window3 = window2
    else:
        window3.extend(window2)
    count += 1
#add windows to df
df2['window'] = window3

# split into training and validation
np.random.seed(seed)
trnotes = np.random.choice(notes_2018_in_cndf, len(notes_2018_in_cndf) * 2 // 3, replace=False)
tenotes = [i for i in notes_2018_in_cndf if i not in trnotes]
trnotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in trnotes]
tenotes = [re.sub("enote_", "", re.sub(".csv", "", i)) for i in tenotes]

#Add 1 to make values 0, 1, 2 (from -1, 0, 1) because -1 is not recognized by to_categorical
#msk = to_categorical(df2[['Msk_prob']].values+1)
train_windows = df2[df2.note.isin(trnotes)]['window']
test_windows = df2[df2.note.isin(tenotes)]['window']
train_labels = df2[df2.note.isin(trnotes)][out_varnames]
test_labels = df2[df2.note.isin(tenotes)][out_varnames]
#make categorical labels in correct tensor shape
tr_labels = []
te_labels = []
for v in out_varnames:
    r = to_categorical(train_labels[[v]].values+1)
    e = to_categorical(test_labels[[v]].values+1)
    tr_labels.append(r)
    te_labels.append(e)

#caseweights
#start with dummies for the outcomes
y_dums = pd.concat([pd.get_dummies(df2[[i]].astype(str)) for i in out_varnames], axis=1)
df_dums = pd.concat([y_dums, df2['note']], axis=1)
#weight non-neutral tokens by the inverse of their prevalence
tr_cw = []
for v in out_varnames:
    non_neutral = np.array(np.sum(y_dums[[i for i in y_dums.columns if ("_0" not in i) and (v in i)]], axis=1)).astype('float32')
    nnweight = 1 / np.mean(non_neutral[df_dums.note.isin(trnotes)])
    caseweights = np.ones(df_dums.shape[0])
    caseweights[non_neutral.astype(bool)] *= nnweight
    tr_caseweights = caseweights[df_dums.note.isin(trnotes)]
    tr_cw.append(tr_caseweights)

#structured data tensors
# first, scale the structured data
scaler = StandardScaler()
scaler.fit(df2[str_varnames].loc[df2.note.isin(trnotes)])
sdf = copy.deepcopy(df2)
sdf[str_varnames] = scaler.transform(df2[str_varnames])
#get training & test data in numpy arrays
train_struc = np.asarray(sdf[sdf.note.isin(trnotes)][str_varnames]).astype('float32')
test_struc = np.asarray(sdf[sdf.note.isin(tenotes)][str_varnames]).astype('float32')

#create vocabulary index
#from keras.preprocessing.text import Tokenizer
#vectorize text (must use venv_ft environment -- not a conda environment, which only allows tensorflow 2.0 on mac)
vectorizer = TextVectorization(output_sequence_length=win_size,
                               standardize=None) #this is CRITICAL -- default will strip '_' and smash multi-word-expressions together
#train_windows_s = tf.data.Dataset.from_tensor_slices(train_windows)
vectorizer.adapt(np.array(train_windows))
#vectorizer.get_vocabulary()
#so now each window is represented by a vector that maps each word with an integer
#first note:
#train_windows[df2.note == list(df2.note.unique())[1]]
#vector mapping:
#vectorizer(train_windows[df2.note == list(df2.note.unique())[1]])
#look at mapping for the first window
# vectorizer.get_vocabulary()[5]
# vectorizer.get_vocabulary()[3]
# vectorizer.get_vocabulary()[2]
# vectorizer.get_vocabulary()[9]
# vectorizer.get_vocabulary()[0]

#vectorize training & test windows
x_train = vectorizer(np.array([[s] for s in train_windows])).numpy()
x_test = vectorizer(np.array([[s] for s in test_windows])).numpy()

#get the vocabulary from our vectorized text
vocab = vectorizer.get_vocabulary()
#make a dictionary mapping words to their indices
word_index = dict(zip(vocab, range(len(vocab))))
#e.g. access the index for the word 'notes'
# word_index['notes']
#check if word is in word index
#'note' in word_index.items()

#next, pick up at "load pre-trained word embeddings"
# https://keras.io/examples/nlp/pretrained_word_embeddings/
# https://rajmak.in/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/
# from gensim.test.utils import datapath
#from gensim.models import Word2Vec
# Load the model
#cr_embed = Word2Vec.load(f"{pretr_embeddingsdir}W2V_300/w2v_OA_CR_300d.bin")
#also possible to directly create an embedding layer, but documentation is not good
#cr_embed.wv.get_keras_embedding(train_embeddings=False).input_dim
# Load a word2vec model stored in the C *binary* format.
#cr_embed = KeyedVectors.load(f"{pretr_embeddingsdir}W2V_300_cr/w2v_OA_CR_300d.bin", mmap='r')
cr_embed = KeyedVectors.load(f"{pretr_embeddingsdir}w2v_oa_all_300d.bin", mmap='r')
#check if word is in vocab
#'note' in cr_embed.wv.vocab
#get word vec for a given word
#cr_embed.wv.word_vec('note')
#create an embedding matrix (embeddings mapped to word indices)
# adding 1 ensures that there is a row of zeros in the embedding matrix for words in the text that are not in the embeddings vocab
num_words = len(word_index) + 1
embedding_len = 300
embedding_matrix = np.zeros((num_words, embedding_len))
#embedding_matrix.shape
#add embeddings for each word to the matrix
hits = 0
misses = 0
for word, i in word_index.items():
    #by default, words not found in embedding index will be all-zeros
    if word in cr_embed.wv.vocab:
        embedding_matrix[i] = cr_embed.wv.word_vec(word)
        hits += 1
    else:
        misses += 1
print("Converted %s words (%s misses)" % (hits, misses))

#load embeddings matrix into an Embeddings layer
cr_embed_layer = Embedding(embedding_matrix.shape[0],
                           embedding_matrix.shape[1],
                           embeddings_initializer=keras.initializers.Constant(embedding_matrix),
                           trainable=False)

#compare hits and misses to df (number should be equal)
# unique_tokens = df[df.note.isin(trnotes)].groupby('token').head(1).copy()
# unique_tokens[(unique_tokens.identity_298 == 0) & (unique_tokens.identity_12 == 0)].shape
# unique_tokens[(unique_tokens.identity_298 != 0) & (unique_tokens.identity_12 != 0)].shape
# missing_df_embed = list(unique_tokens[(unique_tokens.identity_298 == 0) & (unique_tokens.identity_12 == 0)]['token'])
# missing_layer_embed = []
# for word, i in word_index.items():
#     if word not in cr_embed.wv.vocab:
#         if word not in missing_df_embed:
#             missing_layer_embed.append(word)

#test for missing values before training
def test_nan_inf(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nan.')
    if np.isinf(tensor).any():
        raise ValueError('Tensor contains inf.')
test_nan_inf(x_train)
test_nan_inf(x_test)
test_nan_inf(tr_labels)
test_nan_inf(te_labels)
test_nan_inf(train_struc)
test_nan_inf(test_struc)

best_batch_s = 32
epochs = 80
tr_loss_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
#set lists for output
deep_loss = []
deep_val_loss = []
model_name = []
train_sbriers = []
test_sbriers = []

def kerasmodel(n_units, n_lstm, n_dense):
    nlp_input = Input(shape=(win_size,), name='nlp_input')
    meta_input = Input(shape=(len(str_varnames),), name='meta_input')
    x = cr_embed_layer(nlp_input)
    for l in range(n_lstm-1):
        x = Bidirectional(LSTM(n_units, return_sequences=True))(x)
    nlp_out = Bidirectional(LSTM(n_units))(x)
    y = concatenate([nlp_out, meta_input])
    for i in range(n_dense):
        y = Dense(n_units, activation='relu')(y)
    z = Dense(3, activation='sigmoid')(y)
    model = Model(inputs=[nlp_input, meta_input], outputs=[z])
    return(model)

#eventually convert to iterating over model parameters
# m_grid = pd.DataFrame([[64, 1, 1],
#                        [64, 3, 3],
#                        [256, 1, 1],
#                        [256, 3, 3,]])
# m_grid = m_grid.rename(columns=dict({0: 'n_units', 1: 'n_lstm', 2: 'n_dense'}))
#
# for r in m_grid.shape[0]:

#iterate over the frailty aspects
for m in range(len(tr_labels)):
    n_units = 256
    n_lstm = 3
    n_dense = 3
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_bl{n_units}_bl{n_units}_d{n_units}_d{n_units}_d{n_units}_sw"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          sample_weight=tr_cw[m],
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)



for m in range(len(tr_labels)):
    n_units = 256
    n_lstm = 1
    n_dense = 1
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_d{n_units}_sw"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          sample_weight=tr_cw[m],
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)



for m in range(len(tr_labels)):
    n_units = 64
    n_lstm = 1
    n_dense = 1
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_d{n_units}_sw"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          sample_weight=tr_cw[m],
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)


#iterate over the frailty aspects
for m in range(len(tr_labels)):
    n_units = 256
    n_lstm = 3
    n_dense = 3
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_bl{n_units}_bl{n_units}_d{n_units}_d{n_units}_d{n_units}"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)

for m in range(len(tr_labels)):
    n_units = 256
    n_lstm = 1
    n_dense = 1
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_d{n_units}"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)


for m in range(len(tr_labels)):
    n_units = 64
    n_lstm = 1
    n_dense = 1
    frail_lab = out_varnames[m]
    #model name
    mod_name = f"bl{n_units}_d{n_units}"
    fr_mod = f"{frail_lab}_{mod_name}"
    model_2 = kerasmodel(n_units, n_lstm, n_dense)
    model_2.compile(loss='categorical_crossentropy',
                    optimizer=tf.keras.optimizers.Adam(1e-4),
                    metrics=['acc'])
    #fit model
    history = model_2.fit([x_train, train_struc],
                          tr_labels[m],
                          validation_data=([x_test, test_struc], te_labels[m]),
                          epochs=epochs,
                          batch_size=best_batch_s,
                          callbacks=[tr_loss_earlystopping])
    #add loss to list
    deep_loss.append(history.history['loss'])
    deep_val_loss.append(history.history['val_loss'])
    model_name.append(fr_mod)
    #save as df
    tr_m_loss = pd.DataFrame(history.history['loss']).transpose()
    val_m_loss = pd.DataFrame(history.history['val_loss']).transpose()
    index_names = dict({1: fr_mod})
    col_names = dict(zip(range(tr_m_loss.shape[1]), range(1, tr_m_loss.shape[1] + 1)))
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
    tr_sb = pd.DataFrame(tr_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    train_sbriers.append(tr_sb)
    # save sbrier
    tr_sb.to_csv(f"{outdir}{fr_mod}_tr_sbrier.csv")
    #make predictions on testing data
    te_probs = model_2.predict([x_test, test_struc])
    #save predictions
    pd.DataFrame(te_probs).to_csv(f"{outdir}{fr_mod}_val_preds.csv")
    #scaled briers for each class
    te_sb = []
    for i in range(3):
        te_sb.append(scaled_brier(te_labels[m][:, i], te_probs[:, i]))
    te_sb = pd.DataFrame(te_sb).transpose().rename(columns=dict({0: 'neg', 1: 'neut', 2: 'pos'}))
    test_sbriers.append(te_sb)
    # save sbrier
    te_sb.to_csv(f"{outdir}{fr_mod}_te_sbrier.csv")

# early stopping causes differences in epochs -- pad with NA so columns match
train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss]))) * np.nan
val_loss = train_loss.copy()
for i, c in enumerate(deep_loss):
    train_loss[i, :len(c)] = c
train_loss = pd.DataFrame(train_loss)
for i, c in enumerate(deep_val_loss):
    val_loss[i, :len(c)] = c
val_loss = pd.DataFrame(val_loss)
# rename index and columns
index_names = dict(zip((range(len(model_name))), model_name))
col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1] + 1)))
train_loss = train_loss.rename(index=index_names, columns=col_names)
val_loss = val_loss.rename(index=index_names, columns=col_names)
# save
train_loss.to_csv(f"{outdir}{exp}_{mod_name}_train_loss.csv")
val_loss.to_csv(f"{outdir}{exp}_{mod_name}_val_loss.csv")
#combine all sbriers together and save
train_sbrier_out = pd.concat(train_sbriers)
test_sbrier_out = pd.concat(test_sbriers)
train_sbrier_out = train_sbrier_out.rename(index=index_names)
test_sbrier_out = test_sbrier_out.rename(index=index_names)


#
# def expand_grid(grid):
#    return pd.DataFrame([row for row in product(*grid.values())],
#                        columns=grid.keys())
#
# mgrid_dict = {'n_units': [64, 256],
#               'n_lstm': [1, 3],
#               'n_dense': [1, 3]}
#
# expand_grid(mgrid_dict)

#
# #pd.read_csv(f"{outdir}batch_loss_grace.csv")
# best_batch_s = 32
# #models with more epochs
# epochs = 50
# tr_loss_earlystopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
# #set lists for output
# deep_loss = []
# deep_val_loss = []
# model_name = []
# #model name
# mod_name = 'bl256_bl256_bl128_d128_d128_d64'
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# nlp_out = Bidirectional(LSTM(128))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(128, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(1e-4),
#                 metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=epochs,
#                       batch_size=best_batch_s,
#                       callbacks=[tr_loss_earlystopping])
# #add loss to list
# deep_loss.append(history.history['loss'])
# deep_val_loss.append(history.history['val_loss'])
# model_name.append(mod_name)
# #save
# deep_loss.to_csv(f"{outdir}{mod_name}_train_loss.csv")
# deep_val_loss.to_csv(f"{outdir}{mod_name}_val_loss.csv")
#
# #model name
# mod_name = 'bl256_bl256_bl128_d128_d64'
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# nlp_out = Bidirectional(LSTM(128))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(1e-4),
#                 metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=epochs,
#                       batch_size=best_batch_s,
#                       callbacks=[tr_loss_earlystopping])
# #add loss to list
# deep_loss.append(history.history['loss'])
# deep_val_loss.append(history.history['val_loss'])
# model_name.append(mod_name)
# #save
# deep_loss.to_csv(f"{outdir}{mod_name}_train_loss.csv")
# deep_val_loss.to_csv(f"{outdir}{mod_name}_val_loss.csv")
#
# #model name
# mod_name = 'bl256_bl256_bl128_d128'
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# nlp_out = Bidirectional(LSTM(128))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(128, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(1e-4),
#                 metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=epochs,
#                       batch_size=best_batch_s,
#                       callbacks=[tr_loss_earlystopping])
# #add loss to list
# deep_loss.append(history.history['loss'])
# deep_val_loss.append(history.history['val_loss'])
# model_name.append(mod_name)
# #save
# deep_loss.to_csv(f"{outdir}{mod_name}_train_loss.csv")
# deep_val_loss.to_csv(f"{outdir}{mod_name}_val_loss.csv")
#
# #model name
# mod_name = 'bl256_bl128_d128_d128_d64'
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# emb = Bidirectional(LSTM(256, return_sequences=True))(emb)
# nlp_out = Bidirectional(LSTM(128))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(128, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(1e-4),
#                 metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=epochs,
#                       batch_size=best_batch_s,
#                       callbacks=[tr_loss_earlystopping])
# #add loss to list
# deep_loss.append(history.history['loss'])
# deep_val_loss.append(history.history['val_loss'])
# model_name.append(mod_name)
# #save
# deep_loss.to_csv(f"{outdir}{mod_name}_train_loss.csv")
# deep_val_loss.to_csv(f"{outdir}{mod_name}_val_loss.csv")
#
# #model name
# mod_name = 'bl256_d128_d128_d64'
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# nlp_out = Bidirectional(LSTM(256))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(128, activation='relu')(x)
# x = Dense(128, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy',
#                 optimizer=tf.keras.optimizers.Adam(1e-4),
#                 metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=epochs,
#                       batch_size=best_batch_s,
#                       callbacks=[tr_loss_earlystopping])
# #add loss to list
# deep_loss.append(history.history['loss'])
# deep_val_loss.append(history.history['val_loss'])
# model_name.append(mod_name)
# #save
# deep_loss.to_csv(f"{outdir}{mod_name}_train_loss.csv")
# deep_val_loss.to_csv(f"{outdir}{mod_name}_val_loss.csv")
#
# #early stopping causes differences in epochs -- pad with NA so columns match
# train_loss = np.ones((len(deep_loss), np.max([len(e) for e in deep_loss])))*np.nan
# val_loss = train_loss
# for i, c in enumerate(deep_loss):
#     train_loss[i, :len(c)] = c
# train_loss = pd.DataFrame(train_loss)
# for i, c in enumerate(deep_val_loss):
#     val_loss[i, :len(c)] = c
# val_loss = pd.DataFrame(val_loss)
# #rename index and columns
# index_names = dict(zip((range(len(model_name))), model_name))
# col_names = dict(zip(range(train_loss.shape[1]), range(1, train_loss.shape[1]+1)))
# train_loss = train_loss.rename(index=index_names, columns=col_names)
# val_loss = val_loss.rename(index=index_names, columns=col_names)
# #save
# train_loss.to_csv(f"{outdir}{exp}_train_loss.csv")
# val_loss.to_csv(f"{outdir}{exp}_val_loss.csv")
#

#prep for tuning regularization
# best_batch_s = 32
# #models with more epochs
# epochs = 10
# #set lists for output
# deep_loss = []
# deep_val_loss = []
# model_name = []
# def reg_test(bilstm, dense, dropout, reg):
#     #model name
#     mod_name = f"bl{bilstm}_dense{dense}_drop_{dropout}_reg_{reg}"
#     nlp_input = Input(shape=(win_size,), name='nlp_input')
#     meta_input = Input(shape=(len(str_varnames),), name='meta_input')
#     emb = cr_embed_layer(nlp_input)
#     nlp_out = Bidirectional(LSTM(bilstm))(emb)
#     x = concatenate([nlp_out, meta_input])
#     x = Dense(dense, activation='relu')(x)
#     x = Dense(dense/2, activation='relu')(x)
#     x = Dense(dense/4, activation='relu')(x)
#     x = Dense(3, activation='sigmoid')(x)
#     model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
#     model_2.compile(loss='categorical_crossentropy',
#                     optimizer=tf.keras.optimizers.Adam(1e-4),
#                     metrics=['acc'])
#     #fit model
#     history = model_2.fit([x_train, train_struc],
#                           tr_Msk_prob,
#                           validation_data=([x_test, test_struc], te_Msk_prob),
#                           epochs=epochs,
#                           batch_size=best_batch_s)
#     #add loss to list
#     deep_loss.append(history.history['loss'])
#     deep_val_loss.append(history.history['val_loss'])
#     model_name.append(mod_name)
#
# #make dfs and rename index & columns
# deep_loss = pd.DataFrame(np.vstack(deep_loss))
# deep_val_loss = pd.DataFrame(np.vstack(deep_val_loss))
# index_names = dict(zip((range(len(model_name))), model_name))
# col_names = dict(zip(range(epochs), range(1, epochs+1)))
# deep_loss = deep_loss.rename(index=index_names, columns=col_names)
# deep_val_loss = deep_val_loss.rename(index=index_names, columns=col_names)
# #save
# deep_loss.to_csv(f"{outdir}deep_loss.csv")
# deep_val_loss.to_csv(f"{outdir}deep_val_loss.csv")



# #plot loss
# import matplotlib.pyplot as plt
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs = range(1, len(loss) + 1)
# plt.figure()
# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training loss')
# plt.legend()
# plt.show()
#
# #make predictions
# probabilities = model_2.predict([x_test, test_struc])
# #sbrier for neg class
# scaled_brier(te_Msk_prob[:,0], probabilities[:,0])
# #sbrier for neut class
# scaled_brier(te_Msk_prob[:,1], probabilities[:,1])
# #sbrier for pos class
# scaled_brier(te_Msk_prob[:,2], probabilities[:,2])
#
# from keras.utils import plot_model
# plot_model(model_2, show_shapes=True, to_file='/Users/martijac/Documents/Frailty/frailty_classifier/output/n_nets/model.png')
#
# #working model from http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/
# from keras.models import Model
# from keras.layers import Dense, Input, LSTM, Bidirectional, concatenate
# from keras import regularizers
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# nlp_out = Bidirectional(LSTM(128, dropout=0.3, recurrent_dropout=0.3, kernel_regularizer=regularizers.l2(0.01)))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=5,
#                       batch_size=128)
#
# #working model from http://digital-thinking.de/deep-learning-combining-numerical-and-text-features-in-deep-neural-networks/
# from keras.models import Model
# from keras.layers import Dense, Input, LSTM, Bidirectional, concatenate
# from keras import regularizers
# nlp_input = Input(shape=(win_size,), name='nlp_input')
# meta_input = Input(shape=(len(str_varnames),), name='meta_input')
# emb = cr_embed_layer(nlp_input)
# nlp_out = Bidirectional(LSTM(128))(emb)
# x = concatenate([nlp_out, meta_input])
# x = Dense(64, activation='relu')(x)
# x = Dense(3, activation='sigmoid')(x)
# model_2 = Model(inputs=[nlp_input, meta_input], outputs=[x])
# model_2.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# #fit model
# history = model_2.fit([x_train, train_struc],
#                       tr_Msk_prob,
#                       validation_data=([x_test, test_struc], te_Msk_prob),
#                       epochs=5,
#                       batch_size=128)
#
#
# #https://rajmak.in/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/
# from keras.models import Sequential
# from keras.layers import Conv1D, GlobalMaxPooling1D, Flatten
# from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, Bidirectional
# model_1 = Sequential()
# model_1.add(cr_embed_layer)
# model_1.add(Conv1D(250, 3, padding='valid', activation='relu', strides=1))
# model_1.add(GlobalMaxPooling1D())
# model_1.add(Dense(250))
# model_1.add(Dropout(0.2))
# model_1.add(Activation('relu'))
# model_1.add(Dense(3))
# model_1.add(Activation('sigmoid'))
# model_1.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
# model_1.summary()
# model_1.fit(x_train, tr_Msk_prob, validation_data=(x_test, te_Msk_prob), epochs=10, batch_size=128)

#example from https://rajmak.in/2017/12/07/text-classification-classifying-product-titles-using-convolutional-neural-network-and-word2vec-embedding/
# clothing = pd.read_csv("/Users/martijac/Downloads/product-titles-cnn-data/clothing.tsv", sep='\t')
# cameras = pd.read_csv("/Users/martijac/Downloads/product-titles-cnn-data/cameras.tsv", sep='\t')
# home_appliances = pd.read_csv("/Users/martijac/Downloads/product-titles-cnn-data/home.tsv", sep='\t')
# category = pd.concat([clothing['category'], cameras['category'], home_appliances['category']]).values
# category = to_categorical(category)
# from keras.preprocessing.text import Tokenizer
# all_texts = clothing['title'] + cameras['title'] + home_appliances['title']
# all_texts = all_texts.drop_duplicates(keep=False)
# MAX_NB_WORDS = 200000
# tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
# tokenizer.fit_on_texts(all_texts)
# word_index = tokenizer.word_index

# from sklearn.preprocessing import StandardScaler
# # scaling embeddings and structured data
# scaler = StandardScaler()
# scaler.fit(df2[str_varnames + embedding_colnames].loc[df2.note.isin(trnotes)])
# sdf = copy.deepcopy(df2)
# sdf[str_varnames + embedding_colnames] = scaler.transform(df2[str_varnames + embedding_colnames])
#
# seed = 1234
#
# def draw_hps(seed):
#     np.random.seed(seed)
#     hps = (int(np.random.choice(list(range(4, 40)))),  # window size
#            int(np.random.choice(list(range(1, 10)))),  # n dense
#            int(2 ** np.random.choice(list(range(5, 11)))),  # n units
#            float(np.random.uniform(low=0, high=.5)),  # dropout
#            float(10 ** np.random.uniform(-8, -2)),  # l1/l2 penalty
#            bool(np.random.choice(list(range(2)))))  # semipar
#     #model = makemodel(*hps)
#     #return model, hps
#     return hps
#
# hps = draw_hps(seed + mainseed)
#
# def tensormaker(D, notelist, cols, winsize):
#     # take a data frame and a list of notes and a list of columns and a window size and return an array for feeting to tensorflow
#     note_arrays = [np.array(D.loc[D.note == i, cols]) for i in notelist]
#     notelist = []
#     for j in range(len(note_arrays)):
#         lags, leads = [], []
#         for i in range(int(np.ceil(winsize / 2)) - 1, 0, -1):
#             li = np.concatenate([np.zeros((i, note_arrays[j].shape[1])), note_arrays[j][:-i]], axis=0)
#             lags.append(li)
#         assert len(set([i.shape for i in lags])) == 1  # make sure they're all the same size
#         for i in range(1, int(np.floor(winsize / 2)) + 1, 1):
#             li = np.concatenate([note_arrays[j][i:], np.zeros((i, note_arrays[j].shape[1]))], axis=0)
#             leads.append(li)
#         assert len(set([i.shape for i in leads])) == 1  # make sure they're all the same size
#         x = np.squeeze(np.stack([lags + [note_arrays[j]] + leads]))
#         notelist.append(np.swapaxes(x, 1, 0))
#     return np.concatenate(notelist, axis=0)
#
# #create tensor of training data
# # using scaled data, training notes, structured data & embeddings, and a windowsize of 11
# Xtr = tensormaker(sdf, trnotes, str_varnames + embedding_colnames, 11)
#
# #start with one note, 2 pieces of str data and 2 embeddings
# D = sdf
# notelist = trnotes
# cols = str_varnames[0:2] + embedding_colnames[0:2]
# winsize = 11
# i = notelist[1]
# #get note & cols and turn into an array
# note_arrays = [np.array(D.loc[D.note == i, cols])]
# notelist = []
# i = 1
# j = 0
# for j in range(len(note_arrays)):
#     lags, leads = [], []
#     for i in range(int(np.ceil(winsize / 2)) - 1, 0, -1):
#         li = np.concatenate([np.zeros((i, note_arrays[j].shape[1])), note_arrays[j][:-i]], axis=0)
#         lags.append(li)
#     assert len(set([i.shape for i in lags])) == 1  # make sure they're all the same size
#     for i in range(1, int(np.floor(winsize / 2)) + 1, 1):
#         li = np.concatenate([note_arrays[j][i:], np.zeros((i, note_arrays[j].shape[1]))], axis=0)
#         leads.append(li)
#     assert len(set([i.shape for i in leads])) == 1  # make sure they're all the same size
#     x = np.squeeze(np.stack([lags + [note_arrays[j]] + leads]))
#     notelist.append(np.swapaxes(x, 1, 0))
#     return np.concatenate(notelist, axis=0)
#
# # take a data frame and a list of notes and a list of columns and a window size and return an array for feeding to tensorflow
# D = sdf
# notelist = trnotes
# cols = str_varnames + embedding_colnames
# winsize = 11
# note_arrays = [np.array(D.loc[D.note == i, cols]) for i in notelist]
# notelist = []
# for j in range(len(note_arrays)):
#     lags, leads = [], []
#     for i in range(int(np.ceil(winsize / 2)) - 1, 0, -1):
#         li = np.concatenate([np.zeros((i, note_arrays[j].shape[1])), note_arrays[j][:-i]], axis=0)
#         lags.append(li)
#     assert len(set([i.shape for i in lags])) == 1  # make sure they're all the same size
#     for i in range(1, int(np.floor(winsize / 2)) + 1, 1):
#         li = np.concatenate([note_arrays[j][i:], np.zeros((i, note_arrays[j].shape[1]))], axis=0)
#         leads.append(li)
#     assert len(set([i.shape for i in leads])) == 1  # make sure they're all the same size
#     x = np.squeeze(np.stack([lags + [note_arrays[j]] + leads]))
#     notelist.append(np.swapaxes(x, 1, 0))
#     product = np.concatenate(notelist, axis=0)
#
# # tensormaker produces an array with shape(len(notelist)*length of each note), winsize, len(str_varnames + embeddings_colnames))
# # so, each token is represented by a matrix where columns (third axis of the tensor shape) are all of the features (structured & embeddings)
# # and rows (second axis) are each token in the window (so 6 lags and 5 leads for an 11-token window). Note that time axis is the second dimension by convention
# # the first axis of the tensor (listed first in shape) is the "samples dimension." Here, it is
# # the total number of tokens in the training set (notes * notelength). This is also the batch dimension, where you will
# # split up tokens into batches to feed into the model. Overall, it make sense to organize the tensors this way. Each matrix
# # represents a "document" (a window of tokens) that needs to be classified.
# test1 = np.array([[[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]],
#                  [[9, 10, 11, 12],
#                   [13, 14, 15, 16],
#                   [17, 18, 19, 20]]])
# test2 = pd.DataFrame([[[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]],
#                  [[9, 10, 11, 12],
#                   [13, 14, 15, 16],
#                   [17, 18, 19, 20]]])
#
# test1 = np.array([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]])
# test2 = pd.DataFrame([[1, 2, 3, 4],
#                   [5, 6, 7, 8],
#                   [9, 10, 11, 12]])

#
# def make_y_list(y):
#     return [y[:, i * 3:(i + 1) * 3] for i in range(len(out_varnames))]
#
# ytr = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in trnotes]))
# yte = make_y_list(np.vstack([sdf.loc[sdf.note == i, y_dums.columns.tolist()] for i in tenotes]))
#
#
# #neural net will need to end in a softmax layer with size 3. This will output
# #probability of each of the 3 output classes (pos, neg, neutral)
