import os
import pandas as pd

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
    from utils.organization import find_outdir
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

    notes_excluded = [i for i in nle if
                      "_".join(i.split("_")[-2:]) not in uidstr]

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

#
# AL02 and AL01 have discrepancies
#
# batchstring='03'
# ALdir = f"{outdir}/saved_models/AL{batchstring}"
# notes_2018 = [i for i in
#           os.listdir(outdir + "notes_labeled_embedded_SENTENCES/")
#           if '.csv' in i and int(i.split("_")[-2][1:]) < 13]
# cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
# cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
# cndf['month'] = cndf.LATEST_TIME.dt.month + (
#         cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
# # generate 'note' label (used in webanno and notes_labeled_embedded)
# cndf.month = cndf.month.astype(str)
# uidstr = ("m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".csv").tolist()
# # conc_notes_df contains official list of eligible patients
# notes_2018_in_cndf = [i for i in notes_2018 if
#                       "_".join(i.split("_")[-2:]) in uidstr]
# notes_excluded = [i for i in notes_2018 if
#                   "_".join(i.split("_")[-2:]) not in uidstr]
# assert len(notes_2018_in_cndf) + len(notes_excluded) == len(notes_2018)
# notes_2018_in_cndf.sort()
# len([i for i in notes_2018 if 'AL02' in i])
# # remove double-annotated notes
# '''
# 25 Jan 2021:  removing double-annotated notes, and notes from different
# span of same patient.  Move the actual files so they don't get picked up later.
# As such, the code below will only do culling once.
# It will also move notes that were excluded in the July 2020 cull
# '''
# import re
# from _06_preprocessing import hasher
# pids = set([re.sub(".csv", "", i.split("_")[-1]) for i in notes_2018_in_cndf])
# keepers = []
# Nhashed = 0
# for i in pids:
#     notes_i = [j for j in notes_2018_in_cndf if i in j]
#     if len(notes_i) >1:
#         # check and see if there are notes from different batches.  Use the last one if so
#         batchstrings = [k.split("_")[1] for k in notes_i]
#         assert all(["AL" in k for k in batchstrings]), "not all double-coded notes are from an AL round."
#         in_latest_batch = [k for k in notes_i if max(batchstrings) in k]
#         # deal with a couple of manual cases
#         if hasher(i) == 'faef3f1f1a76c57e42f9b35a662656096b4e2dfe15040a61a896b1de06ef1e0a45e61e7e9b26f9282047847854d2d1887d19cbf3041aff2130e102d65243e724':
#             keepers.append(f'enote_AL01_v2_m2_{i}.csv')
#             print(i)
#             Nhashed +=1
#         elif hasher(i) == 'e13eced415697e4f59bcc8e75659dcffa4182a8de44b976d5e1d8160407711d276e38946ec52ec366a3ad92f197d92e8d56c1bd4e9029103d17ac12944cc3bc5':
#             keepers.append(f'enote_AL01_m2_{i}.csv')
#             print(i)
#             Nhashed +=1
#         elif len(in_latest_batch) == 1:
#             keepers.append(in_latest_batch[0])
#         elif len(set([k.split("_")[-2] for k in in_latest_batch]))>1: # deal with different spans
#             spans = [k.split("_")[-2] for k in in_latest_batch]
#             latest_span = [k for k in in_latest_batch if max(spans) in k]
#             assert len(latest_span) == 1
#             keepers.append(latest_span[0])
#         elif any(['v2' in k for k in in_latest_batch]): # deal with the case of the "v2" notes -- an outgrowth of confusion around the culling in July 2020
#             v2_over_v1 = [k for k in in_latest_batch if 'v2' in k]
#             assert len(v2_over_v1) == 1
#             keepers.append(v2_over_v1[0])
#         else:
#             print('problem with culling')
#             breakpoint()
#     else:
#         keepers.append(notes_i[0])
#
# droppers = [i for i in notes_2018_in_cndf if i not in keepers]
# # make a little report on keepers and droppers
# report = 'Keepers:\n'
# for i in range(3):
#     x = [j for j in keepers if f'AL0{str(i)}' in j]
#     report += f"{len(x)} notes in AL0{i}\n"
#
# len([i for i in keepers if 'AL01' in i])
#
# x = [i for i in keepers if "AL0" not in i]
# report += f"{len(x)} notes from initial random batches\n"
# report += f"Total of {len(keepers)} notes keeping\n\n"
# report += "Droppers\n"
# for i in range(3):
#     x = [j for j in droppers if f'AL0{str(i)}' in j]
#     report += f"{len(x)} notes in AL0{i}\n"
#
# x = [i for i in droppers if "AL0" not in i]
# report += f"{len(x)} notes from initial random batches\n"
# report += f"Total of {len(droppers)} notes dropping"
# report += "\n\nHere are the notes that got dropped, and their corresponding keeper:\n-----------------\n"
# for i in droppers:
#     p = re.sub(".csv", "", i.split("_")[-1])
#     k = [j for j in keepers if p in j]
#     report += f"Dropping: {i}\n"
#     report += f"Keeping: {k[0]}\n---------\n"
#
# print(report)
# write_txt(report, f"{ALdir}/keep_drop_report.txt")
# assert (Nhashed == 2) | (len(droppers) == 0)
# assert len(keepers) == len(pids)