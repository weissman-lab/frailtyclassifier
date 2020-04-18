
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
from _99_project_module import nrow, write_txt, ncol
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
    # make sure that all of the separators are the correct number of dashes
    dash_token_indices = [fi.index[i] for i in fi.index if fi.token[i] == '--------------------------------------------------------------']
    # make sure that there is an even number of them
    assert len(dash_token_indices) %2 == 0
    # make sure that they all have the same distance apart
    spanlengths = [dash_token_indices[i] - dash_token_indices[i-1] for i in range(1, len(dash_token_indices),2)]
    if len(list(set(spanlengths))) != 1:
        breakpoint()
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
              aggfuncdict,
              lagorder = 0):  # if idx is not None, this is how many random indices to pull
    # First check and see whether the embeddings object is loaded
    if isinstance(embeddings, str): # load it if it's not
        embeddings = KeyedVectors.load(embeddings, mmap='r')
    assert ("Word2Vec" in type(embeddings).__name__) or ("FastText" in type(embeddings).__name__)
    # now load the file and remove its headers (note concatenation indicators)
    fi = pd.read_pickle(f"{anno_dir + webanno_output}/labels/{file}")
    fi = remove_headers(fi)
    # add metadata
    fi['index'] = fi.index
    fi['note'] = file
    # now figure out which rows to process
    centers = list(range(nrow(fi)))
    # loop through the words and make an embeddings matrix
    Elist = [embeddings_catcher(i, embeddings) for i in fi.token]
    # make a variable that indicates whether the word was found in the vocab
    fi['invocab'] = [1 if len(set(Elist[i]))>1 else 0 for i in range(len(Elist))]
    # loop through the functions and apply them to the list of embeddings
    outlist = []
    # identity
    Emat = np.vstack(Elist)
    outlist.append(pd.DataFrame(data = Emat,
                     index = fi.index.tolist(),
                     columns = ["identity_"+ str(i) for i in range(ncol(Emat))]))
    # lags
    for lag in range(1, lagorder):
        x = np.concatenate([np.vstack(Elist[lag:]),
                            np.zeros((lag, len(Elist[0])))],
                           axis = 0)
        outlist.append(pd.DataFrame(data=x,
                       index=fi.index.tolist(),
                       columns=["lag_"+str(lag)+"_" + str(i) for i in range(ncol(x))]))
    # functions in dictionary
    if aggfuncdict is not None:
        for fname in list(aggfuncdict.keys()):
            # pre-allocate a matrix to take the function's input
            empty = np.zeros((nrow(fi), ncol(Emat)))
            for center in centers:
                window = list(range((center - bandwidth), (center + bandwidth)))
                trwindow = [i for i in window if i >= 0 and i < nrow(fi) and fi.invocab.iloc[i] == 1]
                # trim the kernel, for cases where the window overlaps the edges of the note
                ktrim = np.array([kernel[i] for i in range(len(window)) if window[i] in trwindow])
                empty[center,:] = aggfuncdict['wmean'](Emat[trwindow,:], ktrim)
            outlist.append(pd.DataFrame(data=empty,
                                        index=fi.index.tolist(),
                                        columns=[fname + "_" + str(i) for i in range(ncol(Emat))]))
    # construct output
    output = pd.concat([fi]+outlist, axis = 1)
    return output.reset_index(drop=True)


def makeds(argsdict):
    tuples_for_starmap = [(i,
                          anno_dir,
                          webanno_output,
                          argsdict['bandwidth'],
                          argsdict['kernel'],
                          argsdict['embeddings'],
                          aggfunc,
                          argsdict['lagorder']) for i in argsdict['fi']]
    pool = mp.Pool(argsdict['ncores'])
    ll = pool.starmap(featurize, tuples_for_starmap, chunksize=1)
    pool.close()
    outfile = pd.concat(ll)
    outfile.to_csv(f'{outdir}/test_data_{argsdict["embeddings"].split("/")[-1].split(".")[0]}_bw{argsdict["bandwidth"]}.csv')

# these are functions for putting into the aggfunc dictionary
def wmean(Emat, kernel):
    kernel = kernel/sum(kernel)
    return Emat.T @ kernel

aggfunc = dict(wmean = wmean)

## batch 1 dataset
bandwidth = 5
webanno_output = "frailty_phenotype_batch_1_2020-03-02_1328"
makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
            embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
            kernel=np.ones(bandwidth*2),
            bandwidth=bandwidth,
            ncores=mp.cpu_count(),
            lagorder=2))
os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
          f"{outdir}batch1_data_ft_oa_corp_300d_bw5.csv")
##batch 2
webanno_output = "frailty_phenotype_batch_2_2020-03-02_1325"
makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
            embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
            kernel=np.ones(bandwidth*2),
            bandwidth=bandwidth,
            ncores=mp.cpu_count(),
            lagorder=2))
os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
          f"{outdir}batch2_data_ft_oa_corp_300d_bw5.csv")

##batch 3
webanno_output = "frailty_phenotype_test_batch_1_2020-03-06_1522"
makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
            embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
            kernel=np.ones(bandwidth*2),
            bandwidth=bandwidth,
            ncores=mp.cpu_count(),
            lagorder=2))
os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
          f"{outdir}batch3_data_ft_oa_corp_300d_bw5.csv")

##batch 4
webanno_output = "frailty_phenotype_batch_3_2020-04-18_1444"
makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
            embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
            kernel=np.ones(bandwidth*2),
            bandwidth=bandwidth,
            ncores=mp.cpu_count(),
            lagorder=2))
os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
          f"{outdir}batch4_data_ft_oa_corp_300d_bw5.csv")


# if platform.uname()[1] == "grace":
#     # OA embeddings
#     OA = os.popen("find /proj/cwe/built_models/OA_CR |grep -E 'bin' | grep -v .npy").read().split("\n")
#     # penn
#     uphs = os.popen("find /data/penn_cwe/output/trained_models |grep -E 'wv|ft' | grep -v .npy").read().split("\n")
#     #
# elif platform.uname()[1] == 'PAIR-ADM-010.local':
#     # OA embeddings
#     OA = os.popen("find /Users/crandrew/projects/clinical_word_embeddings |grep -E 'bin' | grep -v .npy").read().split("\n")
#     # penn
#     uphs = os.popen("find /Users/crandrew/projects/pwe/output/trained_models |grep -E 'wv|ft' | grep -v .npy").read().split("\n")
#     #


# bandwidth = 10
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
#                     kernel = norm.pdf(np.linspace(-3, 3, bandwidth * 2)),
#                     bandwidth=bandwidth,
#                     ncores=mp.cpu_count(),
#                     lagorder = 2))
#         print(f"done in {(time.time()-start)/60} minutes")


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
