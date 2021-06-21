import numpy as np
import pandas as pd

'''
This script converts BioClinicalBERT and RoBERTa embeddings saved in numpy
format to .csv format for use in RF and Enet scripts in R.
'''

batches = ['AL01', 'AL02', 'AL03', 'AL04', 'AL05']

# Convert cross-validation folds for training
for batch in batches:
    for r in [1, 2, 3]:
        for f in range(10):
            bioclinicalbert_tr = np.load(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'bioclinicalbert/embeddings_bioclinicalbert_r{r}_f{f}_tr.npy'))
            pd.DataFrame(bioclinicalbert_tr).to_csv(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'embeddings/r{r}_f{f}_tr_bioclinicalbert.csv'))
            bioclinicalbert_va = np.load(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'bioclinicalbert/embeddings_bioclinicalbert_r{r}_f{f}_va.npy'))
            pd.DataFrame(bioclinicalbert_va).to_csv(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'embeddings/r{r}_f{f}_va_bioclinicalbert.csv'))
            roberta_tr = np.load(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'roberta/embeddings_roberta_r{r}_f{f}_tr.npy'))
            pd.DataFrame(roberta_tr).to_csv(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'embeddings/r{r}_f{f}_tr_roberta.csv'))
            roberta_va = np.load(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'roberta/embeddings_roberta_r{r}_f{f}_va.npy'))
            pd.DataFrame(roberta_va).to_csv(
                (f'./output/saved_models/{batch}/processed_data/'
                 f'embeddings/r{r}_f{f}_va_roberta.csv'))


# Convert full training and test set
for batch in batches:
    #full training embeddings data for each batch
    bioclinicalbert_final_train = np.load(
        (f'./output/saved_models/{batch}/processed_data/'
         f'bioclinicalbert/embeddings_bioclinicalbert_final.npy'))
    pd.DataFrame(bioclinicalbert_final_train).to_csv(
        (f'./output/saved_models/{batch}/processed_data/'
         f'full_set/full_bioclinicalbert.csv'))
    # full training embeddings data for each batch
    roberta_final_train = np.load(
        (f'./output/saved_models/{batch}/processed_data/'
         f'roberta/embeddings_roberta_final.npy'))
    pd.DataFrame(roberta_final_train).to_csv(
        (f'./output/saved_models/{batch}/processed_data/'
         f'full_set/full_roberta.csv'))
    #full test set .npy saved in AL01 only (test set is the same for all batches)
    bioclinicalbert_final_test = np.load(
        (f'./output/saved_models/AL01/processed_data/'
         f'bioclinicalbert/embeddings_bioclinicalbert_final_test.npy'))
    pd.DataFrame(bioclinicalbert_final_test).to_csv(
        (f'./output/saved_models/{batch}/processed_data/'
         f'test_set/full_bioclinicalbert.csv'))
    #full test set .npy saved in AL01 only (test set is the same for all batches)
    roberta_final_test = np.load(
        (f'./output/saved_models/AL01/processed_data/'
         f'roberta/embeddings_roberta_final_test.npy'))
    pd.DataFrame(roberta_final_test).to_csv(
        (f'./output/saved_models/{batch}/processed_data/'
         f'test_set/full_roberta.csv'))

