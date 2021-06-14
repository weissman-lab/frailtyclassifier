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


def send_message_to_slack(text, url = None):
    if url is None:
        import warnings
        warnings.warn("The code would like to send you a slack message, but you didn't specify a url")
        return
    from urllib import request, parse
    import json
    post = {"text": "{0}".format(text)}
    try:
        json_data = json.dumps(post)
        req = request.Request(url,
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
        arr.append(np.pad(i, (0, maxlen - len(i)), constant_values=np.nan))
    return np.stack(arr)


def compselect(d, pct, seed=0):
    from sklearn.decomposition import TruncatedSVD
    import numpy as np
    '''function returns the number of components accounting for `pct`% of variance'''
    ncomp = d.shape[1] // 2
    while True:
        pca = TruncatedSVD(n_components=ncomp, random_state=seed)
        pca.fit(d)
        cus = pca.explained_variance_ratio_.sum()
        if cus < pct:
            ncomp += ncomp // 2
        else:
            ncomp = np.where((pca.explained_variance_ratio_.cumsum() > pct) == True)[0].min()
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
    return -np.sum(p * np.log(p) + (1 - p) * np.log(1 - p), axis=1)


def nrow(x):
    return x.shape[0]


def ncol(x):
    return x.shape[1]


def ask_user_password(prompt):
    return getpass.getpass(prompt + ": ")

def create_mssql_connection(username='cranedra', host='clarityprod', database='clarity_snapshot_db', domain='UPHS',
                            port='1433', timeout=600, password=None):
    if password is None:
        password = ask_user_password("PW")
    user=domain+'\\'+username
    return create_engine('mssql+pymssql://{}:{}@{}:{}/{}?timeout={}'.\
                         format(user, password, host, port, database, timeout))


def get_clarity_conn(path_to_clarity_creds = None):
    if path_to_clarity_creds is None:
        print("put your creds in a yaml file somewhere safeish and then rerun this function with the path as argument")
        return
    with open(path_to_clarity_creds) as f:
        creds = yaml.safe_load(f)
        return create_mssql_connection(password=creds['pass'])

def get_res_dict(q, conn):
    res = conn.execute(q)
    data = res.fetchall()
    data_d =  [dict(zip(res.keys(), r)) for r in data]
    return data_d

# function to get data
def get_from_clarity_then_save(query=None, clar_conn=None, save_path=None):
    """function to get data from clarity and then save it, or to pull saved data    """
    # make sure that you're not accidentally saving to the cloud
    if save_path is not None:
        # make sure you're not saving it to box or dropbox
        assert ("Dropbox" or "Box") not in save_path, "don't save PHI to the cloud, you goofus"
    else:
        warnings.warn("you're not saving query output to disk")
    # now get the data
    try:
        db_out = get_res_dict(query, clar_conn)
    except Exception as e:
        print("error:  problem with query or connection")
        return e
    # move it to a df
    df = pd.DataFrame(db_out)
    # save it
    if save_path is not None:
        try:
            df.to_json(save_path)
        except Exception as e:
            return e
    return df

def get_res_with_values(q, values, conn):
    res = conn.execute(q, values)
    data = res.fetchall()
    data_d =  [dict(r.items()) for r in data]
    return data_d

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunk_res_with_values(query, ids, conn, chunk_size=10000, params=None):
    if params is None:
        params = {}
    res = []
    for sub_ids in chunks(ids, chunk_size):
        print('.', end='')
        params.update({'ids': sub_ids})
        res.append(pd.DataFrame(get_res_with_values(query, params, conn)))
    print('')
    return pd.concat(res, ignore_index=True)

def combine_notes(df):
    full_notes = []
    for g, dfi in df.groupby('CSN'):
        full_note = '\n'.join(' '.join(list(dfi.sort_values(['NOTE_ENTRY_TIME', 'NOTE_LINE'])['NOTE_TEXT'])).split('  '))
        row = dfi.iloc[0].to_dict()
        _ = row.pop('NOTE_TEXT')
        _ = row.pop('NOTE_LINE')
        row['NOTE_TEXT'] = full_note
        full_notes.append(row)
        sys.stdout.flush()
    print('')
    return pd.DataFrame(full_notes)

def combine_notes_by_type(df, CSN = "PAT_ENC_CSN_ID", note_type = "IP_NOTE_TYPE"):
    full_notes = []
    for g, dfi in df.groupby(CSN):
        dfis = dfi.sort_values(['NOTE_ENTRY_TIME', 'NOTE_LINE'])[['NOTE_TEXT', note_type, 'NOTE_ENTRY_TIME',
                                                                  'NOTE_ID', 'NOTE_STATUS']]
        full_note = ""
        current_type = "YARR, HERE BE ERRORS IF YOU SCRY THIS IN ANY OUTPUT, MATEY"
        for i in range(nrow(dfis)):
            if current_type != dfis[note_type].iloc[i]: # prepend separator
                full_note += f"\n\n#### {dfis[note_type].iloc[i]}, {dfis['NOTE_ENTRY_TIME'].iloc[i]}, " \
                             f"ID: {dfis['NOTE_ID'].iloc[i]}, " \
                             f"Status: {note_status_mapper(dfis['NOTE_STATUS'].iloc[i])} ####\n"
            current_type = dfis[note_type].iloc[i]
            full_note += '\n'.join(dfis['NOTE_TEXT'].iloc[i].split('  '))
        row = dfi.iloc[0].to_dict()
        _ = row.pop('NOTE_TEXT')
        _ = row.pop('NOTE_LINE')
        _ = row.pop(note_type)
        row['NOTE_TEXT'] = full_note
        full_notes.append(row)
    return pd.DataFrame(full_notes)

def combine_all_notes(df, cohort):
    d = df.sort_values(['NOTE_ID', 'CONTACT_NUM']).drop_duplicates(['NOTE_ID', 'NOTE_LINE'], keep='last')
    d = d.merge(cohort, on='PAT_ENC_CSN_ID', how='left')
    f = combine_notes(d)
    del d
    return f

def make_sql_string(lst, dtype="str", mode = "wherelist"):
    assert dtype in ["str", 'int']
    assert mode in ["wherelist", 'vallist']
    if dtype == "int":
        lst = [str(i) for i in lst]
    if mode == "wherelist":
        if dtype == "str":
            out = "('" + "','".join(lst) + "')"
        elif dtype == "int":
            out = "(" + ",".join(lst) + ")"
    elif mode == "vallist":
        if dtype == "str":
            out = "('"+"'),('".join(lst) + "')"
        elif dtype == "int":
            out = "("+"),(".join(lst) + ")"
    return out



import re
def query_filtered_with_temp_tables(q, fdict, rstring = ""):
    """
    The q is the query
    the fdict contains the info on how to filter, and what the foreign table is
    the rstring is some random crap to append to the filter table when making lots of temp tables through multiprocessing
    """
    base_temptab = """
IF OBJECT_ID('tempdb..#filter_n') IS NOT NULL BEGIN DROP TABLE #filter_n END
CREATE TABLE #filter_n (
  :idname :type NOT NULL,
  PRIMARY KEY (:idname)
);
INSERT INTO #filter_n
    (:idname)
VALUES
:ids;
"""
    base_footer = "join #filter_n on #filter_n.:idname = :ftab.:fkey \n"
    filter_header = ""
    filter_footer = ""
    for i in range(len(fdict)):
        tti = re.sub(":idname", list(fdict.keys())[i], base_temptab)
        dtype = list(set(type(j).__name__ for j in fdict[list(fdict.keys())[i]]['vals']))
        assert len(dtype) == 1
        dtype = dtype[0]
        valstring = make_sql_string(fdict[list(fdict.keys())[i]]['vals'], dtype = dtype, mode = 'vallist')
        tti = re.sub(":ids", valstring , tti)
        if dtype == "str":
            tti = re.sub(":type", "VARCHAR(255)", tti)
        elif dtype == "int":
            tti = re.sub(":type", "INT", tti)
        tti = re.sub("filter_n", f"filter_{i}_{rstring}", tti)
        filter_header += tti

        fi = re.sub(":idname", list(fdict.keys())[i], base_footer)
        fi = re.sub(":fkey", fdict[list(fdict.keys())[i]]['foreign_key'], fi)
        fi = re.sub("filter_n", f"filter_{i}_{rstring}", fi)
        fi = re.sub(":ftab", fdict[list(fdict.keys())[i]]['foreign_table'], fi)
        filter_footer += fi
    outq = filter_header + "\n" + q + "\n" + filter_footer
    return outq


if __name__ == "__main__":
    pass
