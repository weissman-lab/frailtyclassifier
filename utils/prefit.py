import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Masking, Input, Bidirectional, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from gensim.models import KeyedVectors
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2
from utils.constants import ROBERTA_MAX_TOKS
from transformers import RobertaTokenizer, RobertaConfig, TFRobertaModel
from utils.transformers_embedding import BCBEmbedder
import os
import re
import tensorflow as tf


def make_model(emb_path,
               sentence_length,
               meta_shape,
               tags,
               train_sent,
               l1_l2_pen,
               n_units,
               n_dense,
               dropout,
               test_sent=None,  # not used
               ALdir=None,  # not used
               embeddings=None,
               emb_filename = None
               ):
    # instatiate vectorizer
    vectorizer = TextVectorization(max_tokens=None,  # unlimited vocab size
                                   output_sequence_length=sentence_length,
                                   standardize=None)  # this is CRITICAL --
    # default will strip '_' and smash multi-word-expressions together
    vectorizer.adapt(np.array(train_sent.fillna("")))
    cr_embed = KeyedVectors.load(emb_path, mmap='r')
    # get the vocabulary from our vectorized text
    vocab = vectorizer.get_vocabulary()
    # make a dictionary mapping words to their indices
    word_index = dict(zip(vocab, range(len(vocab))))
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

    # make the actual model
    nlp_input = Input(shape=(sentence_length,), name='nlp_input')
    meta_input = Input(shape=(meta_shape,), name='meta_input')
    emb = cr_embed_layer(nlp_input)
    mask = Masking(mask_value=0.)(emb)
    lstm = Bidirectional(LSTM(n_units, kernel_regularizer=l1_l2(l1_l2_pen)))(mask)
    for i in range(n_dense):
        d = Dense(n_units, activation='relu',
                  kernel_regularizer=l1_l2(l1_l2_pen))(lstm if i == 0 else drp)
        drp = Dropout(dropout)(d)
    penultimate = concatenate([drp, meta_input])
    out = [Dense(3, activation='softmax', name=t)(penultimate) for t in tags]

    model = Model(inputs=[nlp_input, meta_input], outputs=out)
    return model, vectorizer


def make_transformers_model(emb_path,
                            sentence_length,
                            meta_shape,
                            tags,
                            train_sent,
                            l1_l2_pen,
                            n_units,
                            n_dense,
                            dropout,
                            ALdir,
                            embeddings,
                            test_sent,
                            emb_filename
                            ):
    '''
    creates or loads the embeddings, and builds a model off of that
    :return: a model, for training.  in place of the vectorizer is a dict with embeddings to feed the model
    '''
    # tr = np.load('/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/saved_models/AL01/processed_data/bioclinicalbert/tr.npy')
    # va = np.load('/Users/crandrew/projects/GW_PAIR_frailty_classifier/output/saved_models/AL01/processed_data/bioclinicalbert/te.npy')
    #
    cond1 = os.path.exists(f"{ALdir}processed_data/{embeddings}/{emb_filename}")
    cond2 = os.path.exists(f"{ALdir}processed_data/{embeddings}/{re.sub('_tr', '_va', emb_filename)}")
    if not cond1 & cond2:
        emb = BCBEmbedder(model_type = embeddings)
        tr = emb(train_sent.tolist())
        va = emb(test_sent.tolist())
        np.save(f"{ALdir}processed_data/{embeddings}/{emb_filename}", tr)
        np.save(f"{ALdir}processed_data/{embeddings}/{re.sub('_tr', '_va', emb_filename)}", va)
    else:
        tr = np.load(f"{ALdir}processed_data/{embeddings}/{emb_filename}")
        va = np.load(f"{ALdir}processed_data/{embeddings}/{re.sub('_tr', '_va', emb_filename)}")

    structured_input = Input(shape=meta_shape, name='inp_str')
    embedding_input = Input(shape=768, name='inp_emb')
    drp = Dropout(dropout)(embedding_input)
    for i in range(n_dense):
        d = Dense(n_units, activation='relu',
                  kernel_regularizer=l1_l2(l1_l2_pen))(drp)
        drp = Dropout(dropout)(d)
    penultimate = concatenate([drp, structured_input])
    out = [Dense(3, activation='softmax', name=t)(penultimate) for t in tags]
    model = Model(inputs=[embedding_input, structured_input], outputs=out)
    return model, dict(tr = tr, va = va)


def make_roberta_model(meta_shape,
                       tags,
                       l1_l2_pen,
                       n_units,
                       n_dense,
                       dropout,
                       average_embeddings_no_train=True,
                       emb_path=None,
                       sentence_length=None,
                       train_sent=None,
                       ):
    '''outputs a tokenizer and a roberta model'''
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    structured_input = Input(shape=meta_shape, name='inp_str')
    input_ids = Input(shape=ROBERTA_MAX_TOKS, dtype=tf.int32, name='inp_ids')
    attention_mask = Input(shape=ROBERTA_MAX_TOKS, dtype=tf.int32, name='inp_attnmask')
    config = RobertaConfig.from_pretrained('roberta-base', output_hidden_states=False)
    encoder = TFRobertaModel.from_pretrained('roberta-base', config=config)
    if average_embeddings_no_train == True:
        embedding = encoder(input_ids,
                            attention_mask=attention_mask,
                            )[0][:, 1:, :]
        eavg = tf.reduce_mean(embedding, axis=1)
        drp = Dropout(dropout)(eavg)
    else:
        embedding = encoder(input_ids,
                            attention_mask=attention_mask,
                            )[0][:, 0, :]
        drp = Dropout(dropout)(embedding)
    for i in range(n_dense):
        d = Dense(n_units, activation='relu',
                  kernel_regularizer=l1_l2(l1_l2_pen))(drp)
        drp = Dropout(dropout)(d)
    penultimate = concatenate([drp, structured_input])
    out = [Dense(3, activation='softmax', name=t)(penultimate) for t in tags]
    model = Model(inputs=[input_ids, attention_mask, structured_input], outputs=out)
    if average_embeddings_no_train == True:
        for n, l in enumerate(model.layers):
            if 'bert' in l.name:
                model.layers[n].trainable = False
    return model, tokenizer


if __name__ == "__main__":
    pass
