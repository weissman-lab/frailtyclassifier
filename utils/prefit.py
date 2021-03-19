

import numpy as np
from tensorflow import keras
from tensorflow.keras.layers import Embedding, Masking, Input, Bidirectional, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from gensim.models import KeyedVectors
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l1_l2


def make_model(emb_path,
               sentence_length,
               meta_shape,
               tags,
               train_sent,
               l1_l2_pen,
               n_units,
               n_dense,
               dropout
               ):
    # instatiate vectorizer
    vectorizer = TextVectorization(max_tokens=None,  # unlimited vocab size
                                   output_sequence_length=sentence_length,
                                   standardize=None)  # this is CRITICAL --
    # default will strip '_' and smash multi-word-expressions together
    vectorizer.adapt(np.array(train_sent.fillna("")))
    cr_embed = KeyedVectors.load(emb_path,mmap='r')
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
    out = [Dense(3, activation='softmax', name = t)(penultimate) for t in tags]

    model = Model(inputs=[nlp_input, meta_input], outputs=out)
    return model, vectorizer
