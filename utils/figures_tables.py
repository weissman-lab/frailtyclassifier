import pandas as pd
import os
from utils.organization import find_outdir, summarize_train_test_split

def table_1_demographics():
    '''
    Finds the directory of the new AL batch and:
     1.) compares structured data to testing and training structured data
     2.) reports all structured data for the new AL batch for manual review
     '''
    # set directories
    outdir = find_outdir()
    # summarize the labeled & embedded training notes and all test notes
    summarize_train_test_split()
    # load notes
    notes_train = pd.read_csv(
        f"{outdir}notes_labeled_embedded_SENTENCES/notes_train_official.csv")

    notes_test = pd.read_csv(
        f"{outdir}notes_labeled_embedded_SENTENCES/notes_test_rough.csv")

    # ACD: THIS IS THE SECTION YOU MAY NEED TO EDIT
    # Please make sure this section checks /notes_output/ for batches labeled 'AL',
    # and then compares them to notes_train to see if they have already been
    # labeled & embedded.
    # check for a new AL batch:
    new_AL_batches = [i for i in os.listdir(f"{outdir}notes_output/")
                      if 'AL_' in i
                      and ('AL' + str(i).split("_")[1]) not in list(notes_train.batch)]

    assert len(new_AL_batches) > 0, "Could not find the new AL batch"
    print(f"New AL batch: {new_AL_batches}")

    # make table with new AL notes
    notes_new_AL = []
    for batch in new_AL_batches:
        notes = [i for i in os.listdir(f"{outdir}notes_output/{batch}") if
                 '.txt' in i]
        PAT_IDs = [str(i).split("_")[-1].split(".")[0] for i in notes]
        months = [int(i.split("_")[-2][1:]) for i in notes]
        label_batch = [str(i).split("_")[0] for i in notes]
        fname = [i for i in notes]
        pat_batch = pd.DataFrame(dict(PAT_ID=PAT_IDs,
                                      month=months,
                                      batch=label_batch,
                                      filename=fname))
        notes_new_AL.append(pat_batch)
    notes_new_AL = pd.concat(notes_new_AL).reset_index(drop=True)

    # load structured data
    strdat = pd.read_csv(f"{outdir}structured_data_merged_cleaned.csv",
                         index_col=0)

    # get structured data for new AL batch
    strdat_new_AL = strdat.merge(notes_new_AL[['PAT_ID', 'month']])
    strdat_new_AL['train_test'] = 'new_AL'
    # summarize new AL batch
    drop_cols = ['sd_', 'MV_', 'train_test']
    drop_cols = [i for i in all_str.columns if
                 any(xi in i for xi in drop_cols)]

    new_AL_str = strdat_new_AL.loc[:, ~all_str.columns.isin(drop_cols)].sort_values(
        by='PAT_ID')

    #merge with prior training & test data
    strdat_train = strdat.merge(notes_train[['PAT_ID', 'month']])
    strdat_train['train_test'] = 'train'
    strdat_test = strdat.merge(notes_test[['PAT_ID', 'month']])
    strdat_test['train_test'] = 'test'
    all_str = pd.concat([strdat_new_AL, strdat_train, strdat_test])

    # restrict columns
    drop_cols = ['PAT_ID', 'month', 'sd_', 'MV_']
    drop_cols = [i for i in all_str.columns if any(xi in i for xi in drop_cols)]
    all_str = all_str.loc[:, ~all_str.columns.isin(drop_cols)].groupby(
        'train_test').mean().T

    return(all_str, new_AL_str)


if __name__ == "__main__":
    pass