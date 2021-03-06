

import os
import traceback
import pandas as pd
from configargparse import ArgParser
import numpy as np
from utils.fit import AL_CV
from utils.misc import expand_grid, send_message_to_slack
from utils.constants import TAGS

pd.options.display.max_rows = 4000
pd.options.display.max_columns = 4000


def main():
    p = ArgParser()
    p.add("-b", "--batchstring", help="the batch number", type=str)
    options = p.parse_args()
    batchstring = options.batchstring
    
    outdir = f"{os.getcwd()}/output/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"

    #Multi-task
    # hyperparameter grid
    hp_grid = {'batchstring': [batchstring],
               'embeddings': ['w2v', 'roberta', 'bioclinicalbert'],
               'n_dense': [1, 2],
               'n_units': [64, 256],
               'dropout': [.1, .5],
               'l1_l2': [0, 1e-4],
               'case_weights': [False],
               'repeat': [1,2,3],
               'fold': list(range(10)),
               }
    hp_grid = expand_grid(hp_grid)
    hp_grid.insert(0, 'index', list(range(hp_grid.shape[0])))

    hp_grid.to_csv(f"{ALdir}hyperparameter_grid.csv")
    
    scram_idx = np.random.choice(hp_grid.shape[0], hp_grid.shape[0], replace = False)
    hp_grid = hp_grid.iloc[scram_idx, :]
    
    for i in range(hp_grid.shape[0]):
        try:
            _ = AL_CV(*tuple(hp_grid.iloc[i]))
        except:
            err = traceback.format_exc()
            send_message_to_slack(f"problem with {hp_grid['index'].iloc[i]}\n{err}")

    # single task
    hp_grid = {'batchstring': [batchstring],
               'embeddings': ['w2v', 'roberta', 'bioclinicalbert'],
               'n_dense': [1, 2],
               'n_units': [64, 256],
               'dropout': [.1, .5],
               'l1_l2': [0, 1e-4],
               'case_weights': [False],
               'repeat': [1,2,3],
               'fold': list(range(10)),
               'tags': TAGS}
    hp_grid = expand_grid(hp_grid)
    hp_grid.insert(0, 'index', list(range(hp_grid.shape[0])))
    hp_grid.to_csv(f"{ALdir}hyperparameter_grid_singletask.csv")

    scram_idx = np.random.choice(hp_grid.shape[0], hp_grid.shape[0], replace=False)
    hp_grid = hp_grid.iloc[scram_idx, :]

    for i in range(hp_grid.shape[0]):
        try:
            _ = AL_CV(*tuple(hp_grid.iloc[i]))
        except:
            err = traceback.format_exc()
            send_message_to_slack(f"problem with {hp_grid['index'].iloc[i]}\n{err}")


if __name__ == "__main__":
    main()
    
    
    