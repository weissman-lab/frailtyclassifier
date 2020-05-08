


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
              # webanno_output,  # the specific webanno object, processed per the function in _05_tokenize_and_label.py
              embeddings,  # this is either the path to the embeddings (in which case they will be loaded) or the filename of the loaded embeddings
              ):  # if idx is not None, this is how many random indices to pull
    # First check and see whether the embeddings object is loaded
    if isinstance(embeddings, str): # load it if it's not
        embeddings = KeyedVectors.load(embeddings, mmap='r')
    assert ("Word2Vec" in type(embeddings).__name__) or ("FastText" in type(embeddings).__name__)
    # now load the file and remove its headers (note concatenation indicators)
    # fi = pd.read_pickle(f"{anno_dir + webanno_output}/labels/{file}")
    fi = pd.read_pickle(f"{anno_dir}/labeled_text/{file}")
    fi = remove_headers(fi)
    # add metadata
    fi['index'] = fi.index
    fi['note'] = file
    # lowercase the words
    fi.token = fi.token.str.lower()
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

    # construct output
    output = pd.concat([fi]+outlist, axis = 1)
    return output.reset_index(drop=True)


def makeds(argsdict):
    tuples_for_starmap = [(i,
                          anno_dir,
                          # webanno_output,
                          argsdict['embeddings'],
                          ) for i in argsdict['fi']]
    pool = mp.Pool(argsdict['ncores'])
    ll = pool.starmap(featurize, tuples_for_starmap, chunksize=1)
    pool.close()
    outfile = pd.concat(ll)
    return outfile
    # outfile.to_csv(f'{outdir}/test_data_{argsdict["embeddings"].split("/")[-1].split(".")[0]}_bw{argsdict["bandwidth"]}.csv')



# loop through the embeddings and make the datasets
if os.uname()[1] != 'PAIR-ADM-010.fios-router.home': # only run this on grace
    OApaths = []
    for currentpath, folders, files in os.walk('/proj/cwe/'):
        for file in files:
            if ('.bin' in file) and ('300' in file):# and (("OA_CR" in file) or ("oa_corp" in file)):
                if '.npy' not in file:
                    OApaths.append(os.path.join(currentpath, file))
    
    UPpaths = []
    for currentpath, folders, files in os.walk('/data/penn_cwe/output/trained_models'):
        for file in files:
            if (('.ft' in file) or ('.wv' in file)) and ('300' in file):# and (("OA_CR" in file) or ("oa_corp" in file)):
                if '.npy' not in file:
                    UPpaths.append(os.path.join(currentpath, file))
    print(OApaths)
    print(UPpaths)
else:
    OApaths = []
    for currentpath, folders, files in os.walk('/Users/crandrew/projects/PWE/'):
        for file in files:
            if ('.bin' in file) and ('300' in file):# and (("OA_CR" in file) or ("oa_corp" in file)):
                if '.npy' not in file:
                    OApaths.append(os.path.join(currentpath, file))

    UPpaths = []
    for currentpath, folders, files in os.walk("/Users/crandrew/projects/PWE/"):
        for file in files:
            if (('.ft' in file) or ('.wv' in file)) and ('300' in file):# and (("OA_CR" in file) or ("oa_corp" in file)):
                if '.npy' not in file:
                    UPpaths.append(os.path.join(currentpath, file))
    print(UPpaths)
    print(OApaths)



paths = OApaths + UPpaths
for i in paths:
    name = i.split("/")[-1]
    name = re.sub('.bin|.wv|.ft', '', name)
    print(name)
    ds = makeds(dict(fi=os.listdir(f'{anno_dir}/labeled_text/'),
                embeddings=i,
                ncores=mp.cpu_count()))
    ds.to_csv(f"{outdir}diagnostic_{name}.csv")

# def main():
#     ## batch 1 dataset
#     bandwidth = 5
#     webanno_output = "frailty_phenotype_batch_1_2020-03-02_1328"
#     makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
#                 embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
#                 kernel=np.ones(bandwidth*2),
#                 bandwidth=bandwidth,
#                 ncores=mp.cpu_count(),
#                 lagorder=2))
#     os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
#               f"{outdir}batch1_data_ft_oa_corp_300d_bw5.csv")
#     ##batch 2
#     webanno_output = "frailty_phenotype_batch_2_2020-03-02_1325"
#     makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
#                 embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
#                 kernel=np.ones(bandwidth*2),
#                 bandwidth=bandwidth,
#                 ncores=mp.cpu_count(),
#                 lagorder=2))
#     os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
#               f"{outdir}batch2_data_ft_oa_corp_300d_bw5.csv")
    
#     ##batch 3
#     webanno_output = "frailty_phenotype_test_batch_1_2020-03-06_1522"
#     makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
#                 embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
#                 kernel=np.ones(bandwidth*2),
#                 bandwidth=bandwidth,
#                 ncores=mp.cpu_count(),
#                 lagorder=2))
#     os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
#               f"{outdir}batch3_data_ft_oa_corp_300d_bw5.csv")
    
#     ##batch 4
#     webanno_output = "frailty_phenotype_batch_3_2020-04-18_1444"
#     makeds(dict(fi=os.listdir(f'{anno_dir}/{webanno_output}/labels'),
#                 embeddings="/Users/crandrew/projects/clinical_word_embeddings/ft_oa_corp_300d.bin",
#                 kernel=np.ones(bandwidth*2),
#                 bandwidth=bandwidth,
#                 ncores=mp.cpu_count(),
#                 lagorder=2))
#     os.system(f"mv {outdir}test_data_ft_oa_corp_300d_bw5.csv "
#               f"{outdir}batch4_data_ft_oa_corp_300d_bw5.csv")

# if __name__ == "__main__":
#     main()