import os
import pandas as pd
from utils.organization import find_outdir

if __name__ == "__main__":
    pass

def find_outdir():
    # get the correct directories
    dirs = ["/Users/martijac/Documents/Frailty/frailty_classifier/output/",
            "/media/drv2/andrewcd2/frailty/output/", "/share/gwlab/frailty/output/"]
    for d in dirs:
        if os.path.exists(d):
            outdir = d
    return(outdir)

def summarize_train_test_split():
    # def table_1_demos():
    outdir = find_outdir()

    # load all notes_labeled_embedded (patients who NOT culled)
    nle = [i for i in os.listdir(f"{outdir}notes_labeled_embedded_SENTENCES/")
           if '.csv' in i]

    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12

    # generate 'note' label (used in webanno and notes_labeled_embedded)
    cndf.month = cndf.month.astype(str)
    uidstr = ("m" + cndf.month.astype(
        str) + "_" + cndf.PAT_ID + ".csv").tolist()

    # conc_notes_df contains official list of eligible patients
    notes_in_cndf = [i for i in nle if
                          "_".join(i.split("_")[-2:]) in uidstr]

    # make df with labels from each note
    nle_pat_id = [str(i).split("_")[-1].split(".")[0] for i in notes_in_cndf]
    nle_month = [int(i.split("_")[-2][1:]) for i in notes_in_cndf]
    nle_batch = [str(i).split("_")[-4] + "_" + str(i).split("_")[-3] for i in notes_in_cndf]
    nle_batch = [str(i).split("_")[1] if 'enote' in i else i for i in nle_batch]
    nle_filename = [i for i in notes_in_cndf]
    nle = pd.DataFrame(dict(PAT_ID=nle_pat_id,
                            month=nle_month,
                            batch=nle_batch,
                            filename=nle_filename))

    # load dropped patients
    dropped = [i for i in
               os.listdir(f"{outdir}notes_labeled_embedded_SENTENCES/dropped_notes/")
               if '.csv' in i]

    dropped_pat_id = [str(i).split("_")[-1].split(".")[0] for i in dropped]
    dropped_month = [int(i.split("_")[-2][1:]) for i in dropped]
    dropped_batch = [str(i).split("_")[-4] + "_" + str(i).split("_")[-3] for i in dropped]
    dropped_batch = [str(i).split("_")[1] if 'enote' in i else i for i in dropped_batch]
    dropped_filename = [i for i in dropped]
    dropped = pd.DataFrame(dict(PAT_ID=dropped_pat_id,
                            month=dropped_month,
                            batch=dropped_batch,
                            filename=dropped_filename))

    # double check that the dropped patients are dropped
    # note, good to check with this merge -- lots of other flawed strategies out there
    nle_keepers = nle.merge(dropped.drop_duplicates(),
                            on=['PAT_ID', 'month', 'batch', 'filename'],
                            how='left',
                            indicator=True)

    nle_keepers = nle_keepers.loc[nle_keepers._merge == 'left_only']\
        .drop(columns=['_merge'])

    # All ingested (curated, labeled, & embedded) training patients and months
    nle_train = nle_keepers.loc[nle_keepers.month < 13]

    # All ingested (curated, labeled, & embedded) test patients and months
    nle_test = nle_keepers.loc[nle_keepers.month > 12]

    # load test notes from notes_output. These notes are "rough" (have not been
    # curated, labeled, or embedded)
    # batch_1 is in notes_output/batch_01 and notes are labeled batch_01
    # batch_2 is in notes_output/batch_02 and notes are labeled batch_02
    # test_batch_1 is in notes_output/batch_03 and notes are labeled batch_03_m[13-24]...
    # batch_3 is in notes_output/batch_04 and notes are labeled batch_03_m[1-11]...
    # test_batch_2 is in notes_output/batch_05 and notes are labeled batch_05
    # batch_6 is in notes_output/batch_06 and notes are labeled batch_6
    # test_batch_3 is in notes_output/batch_07 and notes are labeled batch_07
    # test_batch_4 is in notes_output/batch_08 and notes are labeled batch_08
    # test_batch_5 is in notes_output/batch_09 and notes are labeled batch_09
    # test_batch_6 is in notes_output/batch_10 and notes are labeled batch_10
    # not sure if batch 7 will get curated:
    # test_batch_7 is in notes_output/batch_11 and notes are labeled batch_11
    batches_with_annotated_test_notes = ['batch_01', 'batch_02', 'batch_03',
                                         'batch_05', 'batch_06', 'batch_07',
                                         'batch_08', 'batch_09', 'batch_10']
    notes_output_test = []
    for batch in batches_with_annotated_test_notes:
        notes = os.listdir(f"{outdir}notes_output/{batch}")
        notes = [i for i in notes if 'batch' in i and 'alternate' not in i]
        PAT_IDs = [str(i).split("_")[-1].split(".")[0] for i in notes]
        months = [int(i.split("_")[-2][1:]) for i in notes]
        label_batch = [str(i).split("_")[-4] + "_" + str(i).split("_")[-3] for i in notes]
        fname = [i for i in notes]
        pat_batch = pd.DataFrame(dict(PAT_ID=PAT_IDs,
                                      month=months,
                                      #notes_output_batch=batch,
                                      batch=label_batch,
                                      filename=fname))
        pat_batch = pat_batch[pat_batch.month > 12]
        notes_output_test.append(pat_batch)

    notes_output_test = pd.concat(notes_output_test).reset_index(drop=True)

    # drop batches that are in nle_test (these have been labeled & embedded so we
    # want to keep track of that version, which is the gold standard)
    rough_test_notes = pd.concat(
        [nle_test,
         notes_output_test[~notes_output_test.batch.isin(nle_test.batch)]]
        ).reset_index(drop=True)

    # find duplicates
    complete_dups = rough_test_notes.loc[rough_test_notes.duplicated(
        subset=['PAT_ID', 'month'],
        keep=False)].sort_values(['PAT_ID', 'month'])

    # remove the duplicate in batch_06
    remove_dup = complete_dups[complete_dups.batch == 'batch_06']
    rough_test_notes = rough_test_notes.merge(remove_dup.drop_duplicates(),
                            on=['PAT_ID', 'month', 'batch', 'filename'],
                            how='left',
                            indicator=True)
    rough_test_notes = rough_test_notes.loc[rough_test_notes._merge
                                    == 'left_only'].drop(columns=['_merge'])

    # report number in each batch
    print(rough_test_notes.groupby('batch').agg(test_batch_count = ('PAT_ID', 'count')))
    print(nle_train.groupby('batch').agg(train_batch_count = ('PAT_ID', 'count')))

    rough_test_notes.to_csv(f"{outdir}notes_labeled_embedded_SENTENCES/notes_test_rough.csv")
    nle_train.to_csv(f"{outdir}notes_labeled_embedded_SENTENCES/notes_train_official.csv")