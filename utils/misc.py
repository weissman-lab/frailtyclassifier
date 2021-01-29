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

if __name__ == "__main__":
    pass