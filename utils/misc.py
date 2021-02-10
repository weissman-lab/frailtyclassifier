import os
import pickle
import json

def write_txt(str, path):
    text_file = open(path, "w")
    text_file.write(str)
    text_file.close()

def read_txt(path):
    f = open(path, 'r')
    out = f.read()
    f.close()
    return out

def read_json(path):
    f = open(path, 'r')
    out = f.read()
    dict = json.loads(out)
    f.close()
    return dict

def write_pickle(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_pickle(path):
    with open(path, 'rb') as handle:
        b = pickle.load(handle)
    return b


def sheepish_mkdir(path):
    try:
        os.mkdir(path)
    except FileExistsError:
        pass
    
    
def send_message_to_slack(text):
    from urllib import request, parse
    import json

    post = {"text": "{0}".format(text)}

    try:
        json_data = json.dumps(post)
        req = request.Request("https://hooks.slack.com/services/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",
                              data=json_data.encode('ascii'),
                              headers={'Content-Type': 'application/json'})
        resp = request.urlopen(req)
    except Exception as em:
        print("EXCEPTION: " + str(em))


def expand_grid(grid):
    import pandas as pd
    from itertools import product
    return pd.DataFrame([row for row in product(*grid.values())],
                        columns=grid.keys())

# test for missing values before training
def test_nan_inf(tensor):
    import numpy as np
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nan.')
    if np.isinf(tensor).any():
        raise ValueError('Tensor contains inf.')

def logit(x):
    from numpy import exp
    return 1 / (1 + exp(-x))


def inv_logit(x):
    from numpy import log
    return log(x / (1 - x))

        
def arraymaker(d, x):
    import numpy as np
    maxlen = max([len(i) for i in d[x]])
    arr = []
    for i in d[x]:
        arr.append(np.pad(i, (0,maxlen-len(i)), constant_values = np.nan))
    return np.stack(arr)


def compselect(d, pct, seed=0):
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    '''function returns the number of components accounting for `pct`% of variance'''
    ncomp = d.shape[1]//2
    while True:
        pca = TruncatedSVD(n_components=ncomp, random_state=seed)
        pca.fit(d)
        cus = pca.explained_variance_ratio_.sum()
        if cus < pct:
            ncomp += ncomp//2
        else:
            ncomp = np.where((pca.explained_variance_ratio_.cumsum()>pct) == True)[0].min()
            return ncomp


def hasher(x):
    import hashlib
    hash_object = hashlib.sha512(x.encode('utf-8'))
    hex_dig = hash_object.hexdigest()    
    return hex_dig



def entropy(p):
    '''
    function assumes a square matrix with rows summing to 1
    it returns a vector of entropies
    '''
    import numpy as np
    return -np.sum(p*np.log(p) + (1-p) * np.log(1-p), axis = 1)


if __name__ == "__main__":
    pass



