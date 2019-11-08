
import yaml
from sqlalchemy import create_engine
import warnings
import getpass
import pandas as pd
import sys

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
    except Exception:
        print("error:  problem with query or connection")
        return
    # move it to a df
    df = pd.DataFrame(db_out)
    # save it
    if save_path is not None:
        try:
            df.to_json(save_path)
        except Exception:
            print("error: problem saving the file")
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
    for g, dfi in df.groupby('PAT_ENC_CSN_ID'):
        full_note = '\n'.join(' '.join(list(dfi.sort_values(['NOTE_ENTRY_TIME', 'NOTE_LINE'])['NOTE_TEXT'])).split('  '))
        row = dfi.iloc[0].to_dict()
        _ = row.pop('NOTE_TEXT')
        _ = row.pop('NOTE_LINE')
        row['NOTE_TEXT'] = full_note
        full_notes.append(row)
        sys.stdout.flush()
    print('')
    return pd.DataFrame(full_notes)

def combine_all_notes(df, cohort):
    d = df.sort_values(['NOTE_ID', 'CONTACT_NUM']).drop_duplicates(['NOTE_ID', 'NOTE_LINE'], keep='last')
    d = d.merge(cohort, on='PAT_ENC_CSN_ID', how='left')
    f = combine_notes(d)
    del d
    return f