
'''
This script will contain function that will take text that is tokenized by the function in _05_tokenize_and_label.py
and generate features.
Rather than building features and saving them to static files, it'll be a function that outputs rows of a dataframe.
If a static file is desired, the output can be aggregated.

The function will take the following arguments:
- embeddings:  this is a path to the embeddings to be used.
- bandwidth:  this is the radius of the window around a centroid word
- aggfunc: a dictionary of functions to apply to the matrix of embeddings.  If there is more than one function,
    the results will just get appended to the end of the resultant output
- kernel:  a vector of length bandwidth*2+1 with relative weights for aggregating within the window
- file:  the file to pull observations from.  random if none
- idx: indices to pull.  random if none
- maskfun:  function to apply to the data frame to preprocess it.  It might be used to remove metadata, or change case
'''

import os
import pandas as pd
pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000
import re
import numpy as np
from _99_project_module import nrow, write_txt
from gensim.models import KeyedVectors
from scipy.stats import norm
import matplotlib.pyplot as plt
import inspect

anno_dir = "/Users/crandrew/projects/GW_PAIR_frailty_classifier/annotation/"
webanno_output = "frailty_phenotype_batch_1_2020-02-05_1016"
file = os.listdir(f"{anno_dir+webanno_output}/labels/")[0]
embeddings = '/Users/crandrew/projects/pwe/output/trained_models/ft_d300.ft'
wv = KeyedVectors.load(embeddings, mmap='r')


# take the file and filter it with the masking function
def remove_headers(fi):
    '''
    This removes metadata tags that are bounded by long sequences of dashes
    It'll loop through the tokens and make note of places with long such strings.
    Then it'll check if there is an even number of them.
    If there is an even number, it will define spans.
    Inside each span it'll assert that there is something that looks like a timestamp
    If all that is true, those obs will be removed
    '''
    dash_token_indices = [fi.index[i] for i in fi.index if re.match("-----+", fi.token[i])]
    # make sure that there is an even number of them
    assert len(dash_token_indices) %2 == 0
    # make sure that they all have the same distance apart
    spanlengths = [dash_token_indices[i] - dash_token_indices[i-1] for i in range(1, len(dash_token_indices),2)]
    assert(len(list(set(spanlengths)))) == 1
    # make sure there is at least one year inside these spans
    for i in range(1, len(dash_token_indices),2):
        jstring = "".join(fi.token[(dash_token_indices[i-1]):(dash_token_indices[i]+1)])
        assert re.search("(19[789]\d|20[01]\d)", jstring)
    # if these pass, lose the spans
    for i in range(1, len(dash_token_indices),2):
        fi = fi.drop(index=list(range((dash_token_indices[i-1]-1),(dash_token_indices[i]+2))))
    # now drop the newlines -- they're not in the dictionary
    fi = fi[fi.token != "\n"]
    fi = fi[fi.token != "\n "]
    return fi

# these are functions for putting into the aggfunc dictionary
def wmean(Emat, kernel):
    kernel = kernel/sum(kernel)
    return Emat.T @ kernel

aggfunc = dict(wmean = wmean,
               max = lambda x: np.amax(x, axis=0),
               min = lambda x: np.amin(x, axis=0))


def featurize(file, # the name of the file
              anno_dir, # the location of the annotation output
              webanno_output, # the specific webanno object, processed per the function in _05_tokenize_and_label.py
              bandwidth,  # the bandwidth of the window
              kernel,  # The weights kernel, for weighted functions
              embeddings,  # this is either the path to the embeddings (in which case they will be loaded) or the filename of the loaded embeddings
              aggfuncdict, # this is a dictionary of functions to apply
              indices = None, # the specific indices of the file to process
              howmany = "all"): # if idx is not None, this is how many random indices to pull
    # First check and see whether the embeddings object is loaded
    ### Still haven't gotten this working for fasttext
    if isinstance(embeddings, str): # load it if it's not
        embeddings = KeyedVectors.load(embeddings, mmap='r')
    assert ("Word2Vec" in type(embeddings).__name__) or ("FastText" in type(embeddings).__name__)
    # now load the file and remove its headers (note concatenation indicators)
    fi = pd.read_pickle(f"{anno_dir + webanno_output}/labels/{file}")
    fi = remove_headers(fi)
    # now figure out which rows to process
    if howmany == "all":
        centers = list(range(nrow(fi)))
    elif isinstance(howmany, int) and (indices is None):
        np.random.seed(8675309)
        centers = np.random.choice(nrow(fi), howmany, replace=False)
    else:
        centers = indices
    # loop through the centers and get the rows
    l = []
    for center in centers:
        # instantiate the output dictionary
        outdict = dict(index=fi.index[center],
                       note=file)
        # make a data frame to keep track of the kernel and whether or not words are in the vocab
        # the window is the raw index of the df
        window = list(range((center - bandwidth), (center + bandwidth)))
        # trim the kernel, for cases where the window overlaps the edges of the note
        ktrim = [kernel[i] for i in range(len(window)) if window[i] >= 0 and window[i] < nrow(fi)]
        idx = [i for i in window if i >= 0 and i < nrow(fi)]
        tokdf = pd.DataFrame(fi.token.iloc[idx].str.lower())
        tokdf["k"] = ktrim
        # incovab isn't relevant to fasttext, but it doesn't fail for fasttext either
        tokdf["invocab"] = [1 if embeddings.__contains__(tokdf.token.iloc[i]) else 0 for i in range(nrow(tokdf))]
        # create the embeddings matrix
        Emat = np.vstack([embeddings[tokdf.token.iloc[i]] for i in range(nrow(tokdf)) if tokdf.invocab.iloc[i] == 1])
        # apply the aggregation functions to it
        for i in range(len(aggfuncdict)):
            ki = list(aggfuncdict.keys())[i]
            if 'kernel' in inspect.getfullargspec(aggfuncdict[ki]).args:
                res = aggfuncdict[ki](Emat, tokdf.k[tokdf.invocab == 1])
            else:
                res = aggfuncdict[ki](Emat)
            for j in range(len(res)):
                outdict[ki + "_" + str(j)] = res[j]
        outframe = pd.DataFrame(outdict, index=[0])
        mm = fi.merge(outframe, left_index=True, right_on='index', copy=False)
        l.append(mm)
    return pd.concat(l).reset_index(drop=True)

x = featurize(file = "batch_01_m10_Z1950630_labels.pkl",
              anno_dir = anno_dir,
              webanno_output = webanno_output,
              bandwidth=20,
              kernel = norm.pdf(np.linspace(-3,3,20*2)),
              embeddings = wv,
              aggfuncdict=aggfunc,
              howmany="all")

x.to_csv(f'{os.getcwd()}/output/foo.csv')



# fi = remove_headers(fi)
# fi.head()
#
# # pick a center word
# center = np.random.choice(nrow(fi))
# # instantiate the output dictionary
# outdict = dict(index = fi.index[center],
#                note = file)
# # make a data frame to keep track of the kernel and whether or not words are in the vocab
# # the window is the raw index of the df
# window = list(range((center-bandwidth),(center+bandwidth)))
# # trim the kernel, for cases where the window overlaps the edges of the note
# ktrim = [kernel[i] for i in range(len(window)) if window[i]>=0 and window[i]< nrow(fi)]
#
# idx = [i for i in window if i >= 0 and i < nrow(fi)]
# tokdf = pd.DataFrame(fi.token.iloc[idx].str.lower())
# tokdf["k"] = ktrim
# tokdf["invocab"] = [1 if wv.__contains__(tokdf.token.iloc[i]) else 0 for i in range(nrow(tokdf))]
# # create the embeddings matrix
# Emat = np.vstack([wv[tokdf.token.iloc[i]] for i in range(nrow(tokdf)) if tokdf.invocab.iloc[i] == 1])
# # apply the aggregation functions to it
# for i in range(len(aggfunc)):
#     ki = list(aggfunc.keys())[i]
#     if 'kernel' in inspect.getfullargspec(aggfunc[ki]).args:
#         res = aggfunc[ki](Emat, tokdf.k[tokdf.invocab == 1])
#     else:
#         res = aggfunc[ki](Emat)
#     for j in range(len(res)):
#         outdict[ki+"_"+str(j)] = res[j]
#
# outframe = pd.DataFrame(outdict, index = [0])
# mm = fi.merge(outframe, left_index = True, right_on = 'index', copy = False)
#
#
#
# '''
# Picking up from ACD pulling files from webanno & converting JSON file into labels
# This script will tokenize the source documents & create a word embedding matrix
# '''
#
# import pandas as pd
# from _99_project_module import read_txt, read_json, process_webanno_output
# import os
# import re
# import spacy
# import gensim
# from gensim.models import Word2Vec
#
# pd.options.display.max_rows = 4000
# pd.options.display.max_columns = 4000
#
# anno_dir = f'{os.getcwd()}/annotation/'
# os.listdir(anno_dir)
#
#
# # unzip the output file
# webanno_output = 'ACD_sandbox_2020-01-27_1007.zip'
# annotator_of_record = 'andrew'
# webanno_unzipped_dir = re.sub('\.zip', "", webanno_output)
# test_dir = re.sub('/labels', "", anno_dir+webanno_unzipped_dir)
# if webanno_unzipped_dir not in os.listdir(anno_dir):
#     process_webanno_output(anno_dir, webanno_output)
#     os.system(f"mkdir {anno_dir+webanno_unzipped_dir}/labels/")
#
#
# # load the spacy stuff
# # note: may need to load spacy model en with this script in terminal: [full path to python interpreter] -m spacy download en
# nlp = spacy.load("en", disable=['parser', 'tagger', 'ner'])
#
#
# # stepping through the files, use the spacy nlp function to build a data frame of tokens and their spans
# tags = ['Functionalimpairment', "Msk_prob", "Nutrition", "Resp_imp", 'Fall_risk']
# mapping_dict = dict(frailty_nos = "Functionalimpairment",
#                     msk_prob_tags = "Msk_prob",
#                     nutrition = "Nutrition",
#                     resp_imp_tags = "Resp_imp",
#                     fall_risk_tags = "Fall_risk")
#
# print(anno_dir) #/Users/jmartin89/Documents/Frailty/frailty_classifier/annotation/
# print(webanno_unzipped_dir) #ACD_sandbox_2020-01-27_1007
# for stub in os.listdir(anno_dir+webanno_unzipped_dir+"/annotation"):
#     # get the annotation file and process it into something that I can work with
#     anno = read_json(f"{anno_dir+webanno_unzipped_dir}/annotation/{stub}/{annotator_of_record}.json")
#     l = []
#     for i in tags:
#         try:
#             l.append(pd.DataFrame(anno["_views"]["_InitialView"][i]))
#         except Exception:
#             pass
#     tag_df = pd.concat(l, sort = True)
#     tag_df.drop(columns='sofa', inplace=True)
#     tag_df = pd.melt(tag_df, id_vars=['begin', 'end'])
#     tag_df = tag_df[~tag_df.value.isnull()]
#     tag_df.value = [1 if 'yes' in i else -1 for i in tag_df.value]
#     # get the original note and tokenize it
#     note = read_txt(f"{anno_dir + webanno_unzipped_dir}/source/{stub}")
#     res = nlp(note)
#     span_df = pd.DataFrame([{"token": i.text, 'length': len(i.text_with_ws)} for i in res])
#     span_df['end'] = span_df.length.cumsum().astype(int)
#     span_df['start'] = span_df.end - span_df.length
#     assert int(span_df.end[-1:]) == len(note)
#     # merge them on
#     for i in tags:
#         span_df[i] = 0
#     for i in range(tag_df.shape[0]):
#         var = mapping_dict[tag_df.variable.iloc[i]]
#         span_df.loc[(span_df.end > tag_df.begin.iloc[i]) &
#                     (span_df.start < tag_df.end.iloc[i]), var] = tag_df.value.iloc[i]
#     outfn = re.sub('\.txt', '_labels.pkl', stub)
#     span_df.to_pickle(f"{anno_dir + webanno_unzipped_dir}/labels/{outfn}")
#
# #convert gensim word2vec into format usable by spacy
# from gensim.models import Word2Vec, keyedvectors
# word2vec = Word2Vec.load('/Users/jmartin89/Documents/Frailty/JMworkingproj/W2V_300/w2v_OA_CR_300d.bin')
# word2vec.wv.save_word2vec_format('w2v_test.txt')
# # code for terminal: python -m spacy init-model en /Users/jmartin89/Documents/Frailty/frailty_classifier/spacy/w2v_300 --vectors-loc /Users/jmartin89/Documents/Frailty/frailty_classifier/w2v_test.txt
# word2vec_load = spacy.load('/Users/jmartin89/Documents/Frailty/frailty_classifier/spacy/w2v_300')
# #load the note from above with word2vec as language
# w2v_note = word2vec_load(note)
# #test print the tokens
# for token in w2v_note:
#     print(token)
# #test print the token vectors
# for token in w2v_note:
#     print('Vector for %s:' % token, token.vector)
#
#
#
#
#
#
# '''
# Define a span in terms of token length or span length.
# For each span, construct a one-hot-encoded vector of it.
# Do the semanic vector enrichment, but implement it as an option.
# Each one-hot will be TxV.  Word embeddings take that to TxD.  Need to get it to 1xD.
# There can be multiple 1xD transformations.  Some of those transformations can be dependent on the TFIDF probability or whatever
# It's also the case that there is parameter vector of dimension Tx1 such that I can take transposed embedded text and
# take it to 1xD.  That would effectively be a windower.
# '''
#
# from gensim.models import Word2Vec, keyedvectors
# import numpy as np
# word2vec = Word2Vec.load('/Users/crandrew/projects/pwe/output/trained_models/w2v_d100.wv')
# 1+1
#
# np.sum((word2vec['blood'] - word2vec['plasma'])**2)
#
# def euc(x, y):
#     return (np.sum((x-y)**2))**.5
#
# euc(word2vec['pulmonary'], word2vec['lung'])
#
#
# b = np.array([.1,.8, .1])
#
#
# '''
# embeddings:
#
# 1xd * dxv
# basically SEVR is nust a weighted sum of the embeddings!
# '''