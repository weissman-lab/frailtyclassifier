
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
import sys
import multiprocessing as mp
import platform
import time

outdir = f"{os.getcwd()}/output/"
anno_dir = f"{os.getcwd()}/annotation/"

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
        assert re.search("(19[789]\d|20[012]\d)", jstring)
    # if these pass, lose the spans
    for i in range(1, len(dash_token_indices),2):
        fi = fi.drop(index=list(range((dash_token_indices[i-1]-1),(dash_token_indices[i]+2))))
    # now drop the newlines -- they're not in the dictionary
    fi = fi[fi.token != "\n"]
    fi = fi[fi.token != "\n "]
    return fi


def embeddings_catcher(tok, embeddings):
    '''
    This function is a workaround for the fact that some OOV words throw errors in the OA corpus, because the
    dictionary wasn't identical to the corpus.
    Tok is a string and embeddings is a gensim object
    '''
    try:
        return embeddings[tok]
    except Exception:
        return np.zeros(embeddings.vector_size)


def featurize(file,  # the name of the token/label file
              anno_dir,  # the location of the annotation output
              webanno_output,  # the specific webanno object, processed per the function in _05_tokenize_and_label.py
              bandwidth,  # the bandwidth of the window
              kernel,  # The weights kernel, for weighted functions
              embeddings,  # this is either the path to the embeddings (in which case they will be loaded) or the filename of the loaded embeddings
              aggfuncdict,  # this is a dictionary of functions to apply
              indices,  # the specific indices of the file to process
              howmany):  # if idx is not None, this is how many random indices to pull
    # First check and see whether the embeddings object is loaded
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
        Emat = np.vstack([embeddings_catcher([tokdf.token.iloc[i]], embeddings) if tokdf.invocab.iloc[i] == 1
                          else np.zeros(embeddings.vector_size) for i in range(nrow(tokdf))])
        # Emat = np.vstack([embeddings[tokdf.token.iloc[i]] if tokdf.invocab.iloc[i] == 1
        #                   else np.zeros(embeddings.vector_size) for i in range(nrow(tokdf))])
        # apply the aggregation functions to it
        for i in range(len(aggfuncdict)):
            ki = list(aggfuncdict.keys())[i]
            if 'kernel' in inspect.getfullargspec(aggfuncdict[ki]).args:
                res = aggfuncdict[ki](Emat, tokdf.k)
            else:
                res = aggfuncdict[ki](Emat)
            for j in range(len(res)):
                outdict[ki + "_" + str(j)] = res[j]
        outframe = pd.DataFrame(outdict, index=[0])
        mm = fi.merge(outframe, left_index=True, right_on='index', copy=False)
        assert nrow(mm) == 1
        l.append(mm)
    return pd.concat(l).reset_index(drop=True)


def makeds(argsdict):
    tuples_for_starmap = [(i,
                          anno_dir,
                          webanno_output,
                          argsdict['bandwidth'],
                          argsdict['kernel'],
                          argsdict['embeddings'],
                          aggfunc,
                          None,
                          "all") for i in argsdict['fi']]
    pool = mp.Pool(argsdict['ncores'])
    ll = pool.starmap(featurize, tuples_for_starmap, chunksize=1)
    pool.close()
    outfile = pd.concat(ll)
    outfile.to_csv(f'{outdir}/test_data_{argsdict["embeddings"].split("/")[-1].split(".")[0]}_bw{argsdict["bandwidth"]}.csv')

# these are functions for putting into the aggfunc dictionary
def wmean(Emat, kernel):
    kernel = kernel/sum(kernel)
    return Emat.T @ kernel

def identity(x):
    return x[(nrow(x)//2),:]

def lag1(x):
    return x[(nrow(x)//2-1),:]

def lag2(x):
    return x[(nrow(x)//2-2),:]

aggfunc = dict(identity = identity,
               lag1 = lag1,
               lag2 = lag2,
               wmean = wmean)

webanno_output = "frailty_phenotype_batch_1_2020-02-17_1147"

if platform.uname()[1] == "grace":
    # OA embeddings
    OA = os.popen("find /proj/cwe/built_models/OA_CR |grep -E 'bin' | grep -v .npy").read().split("\n")
    # penn
    uphs = os.popen("find /data/penn_cwe/output/trained_models |grep -E 'wv|ft' | grep -v .npy").read().split("\n")
    #
elif platform.uname()[1] == 'PAIR-ADM-010.local':
    # OA embeddings
    OA = os.popen("find /Users/crandrew/projects/clinical_word_embeddings |grep -E 'bin' | grep -v .npy").read().split("\n")
    # penn
    uphs = os.popen("find /Users/crandrew/projects/pwe/output/trained_models |grep -E 'wv|ft' | grep -v .npy").read().split("\n")
    #


bandwidth = 5
Efiles = [i for i in OA + uphs if len(i) > 0]
print(Efiles)
print(len(Efiles))
# remove if already done:
# BW30
for e in Efiles:
    print(e)
    if f'test_data_{e.split("/")[-1].split(".")[0]}_bw{bandwidth}.csv' not in outdir:
        start = time.time()
        makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
                    embeddings=e,
                    kernel = np.ones(bandwidth*2),
                    bandwidth=bandwidth,
                    ncores=mp.cpu_count()))
        print(f"done in {(time.time()-start)/60} minutes")




# bandwidth = 30
# Efiles = [i for i in OA + uphs if len(i) > 0]
# print(Efiles)
# print(len(Efiles))
# # remove if already done:
# # BW30
# for e in Efiles:
#     print(e)
#     if f'test_data_{e.split("/")[-1].split(".")[0]}_bw{bandwidth}.csv' not in outdir:
#         start = time.time()
#         makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
#                     embeddings=e,
#                     kernel = norm.pdf(np.linspace(-3, 3, argsdict['bandwidth'] * 2)),
#                     bandwidth=bandwidth,
#                     ncores=mp.cpu_count()))
#         print(f"done in {(time.time()-start)/60} minutes")
#
#
#
