from collections import defaultdict
import getpass
import numpy as np
import pandas as pd
import re

from pymongo import MongoClient
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sqlalchemy import create_engine
from sqlalchemy.exc import ProgrammingError, OperationalError
import yaml

class EstimatorSelectionHelper:

    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}

    def fit(self, X, y, cv=3, n_jobs=3, verbose=1, scoring=None, refit=False, kind='grid', *args, **kwargs):
        for key in self.keys:
            model = self.models[key]
            params = self.params[key]
            if type=='grid':
                gs = GridSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                                  verbose=verbose, scoring=scoring, refit=refit,
                                  return_train_score=True, *args, **kwargs)
                print("Running GridSearchCV for %s." % key)
            else:
                gs = RandomizedSearchCV(model, params, cv=cv, n_jobs=n_jobs,
                                  verbose=verbose, scoring=scoring, refit=refit,
                                  return_train_score=True, *args, **kwargs)
                print("Running RandomizedSearchCV for %s." % key)
            gs.fit(X,y)
            self.grid_searches[key] = gs    
            self.scoring = scoring

    def score_summary(self, sort_by=None):
        def row(key, scores, params, kind):
            d = {
                 'estimator': key,
                 'min_{}'.format(kind): min(scores),
                 'max_{}'.format(kind): max(scores),
                 'mean_{}'.format(kind): np.mean(scores),
                 'std_{}'.format(kind): np.std(scores),
            }
            return pd.Series({**params,**d})
    
        result_dict = defaultdict(dict)
        scoring = self.scoring
        if len(scoring) == 1:
            scoring = ['score']
        if sort_by is None:
            sort_by = 'mean_{}'.format(scoring[0])
        for k in self.grid_searches:
            print(k)
            this_one = {}
            for score in scoring:
                params = self.grid_searches[k].cv_results_['params']
                scores = []
                for i in range(self.grid_searches[k].cv):
                    key = "split{}_test_{}".format(i, score)
                    r = self.grid_searches[k].cv_results_[key]
                    scores.append(r.reshape(len(params),1))
    
                all_scores = np.hstack(scores)
                for p, s in zip(params,all_scores):
                    result_dict[tuple([('estimator', k)] + [(k, v) for k, v in p.items()])].update(row(k, s, p, score))
        
        df = pd.DataFrame(list(result_dict.values())).sort_values([sort_by], ascending=False)
    
        columns = ['estimator'] + [c for c in df.columns if re.search('(min|mean|max|std)_', c)]#'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
    
        return df[columns]


from sklearn import metrics
from scipy import stats

def beta_errors(num, denom):
    return stats.beta.interval(.95, num+1, denom-num+1)


def calibration_curve_error_bars(a, p, n_bins=10):
    pmin, pmax = p.min(), p.max()
    binstarts = np.linspace(pmin, pmax, n_bins+1)
    bincentres = binstarts[:-1] + (binstarts[1] - binstarts[0])/2.0
    numerators = np.zeros(n_bins)
    denomonators = np.zeros(n_bins)
    for b in range(n_bins):
        idx_bin = (p >= binstarts[b]) & (p < binstarts[b+1])
        denomonators[b] = idx_bin.sum()
        numerators[b] = a[idx_bin].sum()

    errors = beta_errors(numerators, denomonators)
    return bincentres, numerators, denomonators, errors


def plot_calibration_curve_error_bars(a, p, n_bins=10):
    x, n, d, err = calibration_curve_error_bars(a, p, n_bins)
    fig, ax = plt.subplots(1, 1)
    ax.errorbar(x, n/d, yerr=[n/d-err[0], err[1]-n/d])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of positives")
    ax.set_xlabel("Mean predicted value")
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")


def calc_metrics(y, preds_proba, thresh=0.5):
    m = {}
    preds = preds_proba[:, 1] > thresh
    cm = metrics.confusion_matrix(y, preds)
    m['proportion +'] = preds.mean()
    m['confusion_matrix'] = cm
    m['Accuracy'] = metrics.accuracy_score(y, preds)
    m['F1 score'] = (metrics.f1_score(y, preds))
    m['FP'] = (cm[0, 1]/(cm[:, 1].sum()*1.0))
    m['FN'] = (cm[1, 0]/(cm[:, 0].sum()*1.0))
    m['Specificity'] = (cm[0, 0]/(cm[0, :].sum()*1.0))
    m['Sensitivity'] = (cm[1, 1]/(cm[1, :].sum()*1.0))
    m['PPV'] = (cm[1, 1]/(cm[:, 1].sum()*1.0))
    m['NPV'] = (cm[0, 0]/(cm[:, 0].sum()*1.0))
    fpr, tpr, thresholds = metrics.roc_curve(y, preds_proba[:, 1])
    m['AUC'] = metrics.auc(fpr, tpr)
    return m


def plot_metrics(y, preds, thresh_range=(0, 1), ax=None):
    m_list = []
    for tr in np.linspace(thresh_range[0], thresh_range[1]):
        m = calc_metrics(y, preds, tr)
        m.update({'threshold': tr})
        m_list.append(m)

    m_df = pd.DataFrame(m_list)

    for metric in ['Sensitivity',
                   'Specificity',
                   'PPV',
                   'F1 score',
                   'proportion +']:  # ,'FN','FP'
        ax.plot(m_df['threshold'], m_df[metric], label=metric)

    ax.legend(bbox_to_anchor=(1.35, 1.0))
    return m_df


def plt_auc(pred, actual, ax):
    fpr, tpr, thresholds = metrics.roc_curve(actual, pred[:, 1])
    auc = metrics.auc(fpr, tpr)
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], '--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.text(0.7, 0.2, 'AUC = %0.2f' % (auc))

def binary_cv_metrics(y, preds, m):
    ACC = metrics.accuracy_score(y,preds)
    cm = metrics.confusion_matrix(y,preds)

    m['confusion_matrix'] = cm
    m['Accuracy'] = ACC
    m['F1 score'] = metrics.f1_score(y,preds)
    m['FPR'] = cm[0,1]/(cm[0,:].sum()*1.0)
    m['FNR'] = cm[1,0]/(cm[1,:].sum()*1.0)
    m['Specificity (TNR)'] = cm[0,0]/(cm[0,:].sum()*1.0)
    m['Sensitivity (TPR, Recall)'] = cm[1,1]/(cm[1,:].sum()*1.0)
    m['PPV (Precision)'] = cm[1,1]/(cm[:,1].sum()*1.0)
    m['NPV'] = cm[0,0]/(cm[:,0].sum()*1.0)

def cv_metrics(y,probas,m,threshold=0.5):
    '''Collect performance metrics'''
    preds = probas >= threshold

    fpr, tpr, thresholds = metrics.roc_curve(y, probas)
    AUC = metrics.auc(fpr, tpr)
    m['AUC'] = AUC
    binary_cv_metrics(y, preds, m)


"""Read a password from the console."""
def ask_user_password(prompt):
    return getpass.getpass(prompt + ": ")

def create_mssql_connection(username='beckermi', host='clarityprod', database='clarity_snapshot_db', domain='UPHS', port='1433', timeout=600, password=None):
    if password is None:
        password = ask_user_password("PW")
    user=domain+'\\'+username
    return create_engine('mssql+pymssql://{}:{}@{}:{}/{}?timeout={}'.format(user, password, host, port, database, timeout))

def get_res_dict(q, conn):
    res = conn.execute(q)
    data = res.fetchall()
    data_d =  [dict(zip(res.keys(), r)) for r in data]
    return data_d

def get_res_with_values(q, values, conn):
    res = conn.execute(q, values)
    data = res.fetchall()
    data_d =  [dict(r.items()) for r in data]
    return data_d

def get_clarity_conn():
    with open('clarity_creds.yaml') as f:
        creds = yaml.safe_load(f)
        return create_mssql_connection(password=creds['pass'])

def get_mongo_conn():
    with open('mongo_creds.yaml') as f:
        creds = yaml.safe_load(f)
    client = MongoClient(host=['uphsvlndc116.uphs.upenn.edu', 'uphsvlndc117.uphs.upenn.edu'], port=27017, replicaset='pennsignalsdb')
    is_authed = client.admin.authenticate(creds['user'],creds['pass'])
    if is_authed:
        return client

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
