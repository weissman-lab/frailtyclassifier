import pandas as pd
import os
from utils.organization import find_outdir


def table_1_demographics():
    '''
    Finds the directory of the new AL batch and:
     1.) compares structured data to testing and training structured data
     2.) reports all structured data for the new AL batch for manual review
     '''
    from utils.organization import summarize_train_test_split
    # set directories
    outdir = find_outdir()
    # summarize the labeled & embedded training notes and all test notes
    summarize_train_test_split()
    # load notes
    notes_train = pd.read_csv(
        f"{outdir}notes_labeled_embedded_SENTENCES/notes_train_official.csv")

    notes_test = pd.read_csv(
        f"{outdir}notes_labeled_embedded_SENTENCES/notes_test_rough.csv")

    # Check for a new AL batch: pull all rounds of AL from /saved_models/ then
    # compare to notes_train to see if they have already been labeled & embedded.
    new_AL_batches = [i for i in os.listdir(f"{outdir}saved_models/")
                      if 'AL0' in i
                      and '.txt' not in i
                      and i not in [i.split("_")[0] for i in notes_train.batch]]

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
    drop_cols = [i for i in strdat_new_AL.columns if
                 any(xi in i for xi in drop_cols)]
    new_AL_str = strdat_new_AL.loc[:, ~strdat_new_AL.columns.isin(drop_cols)].sort_values(
        by='PAT_ID')

    #merge with prior training & test data
    strdat_train = strdat.merge(notes_train[['PAT_ID', 'month']])
    strdat_train['train_test'] = 'train'
    strdat_test = strdat.merge(notes_test[['PAT_ID', 'month']])
    strdat_test['train_test'] = 'test'
    all_str = pd.concat([strdat_new_AL, strdat_train, strdat_test])

    # check if new AL batch contains any duplicates (ok in training data if
    # months are different, but not ideal)
    dups = all_str[all_str.PAT_ID.isin([i for i in list(all_str.PAT_ID)
                                 if list(all_str.PAT_ID).count(i) > 1
                                 and i in list(strdat_new_AL.PAT_ID)])].shape[0]
    assert dups < 1, "duplicates in the new AL batch"
    print(f"{dups} duplicates in the new AL batch")

    # restrict columns
    drop_cols = ['PAT_ID', 'month', 'sd_', 'MV_']
    drop_cols = [i for i in all_str.columns if any(xi in i for xi in drop_cols)]
    all_str = all_str.loc[:, ~all_str.columns.isin(drop_cols)].groupby(
        'train_test').mean().T

    #write out
    all_str.to_csv(
        f"{outdir}saved_models/{new_AL_batches[0]}/{new_AL_batches[0]}_StrucData_compare.csv")
    new_AL_str.to_csv(
        f"{outdir}saved_models/{new_AL_batches[0]}/{new_AL_batches[0]}_StrucData_summary.csv")

    return(all_str, new_AL_str)


def lossplot(d, meanlen, d_final = None):
    import matplotlib.pyplot as plt
    from utils.misc import arraymaker
    import numpy as np
    fig = plt.figure(constrained_layout=True, figsize=(10,4))
    gs = fig.add_gridspec(2, 4)
    # overall
    f_ax1 = fig.add_subplot(gs[:2, :2])
    L = arraymaker(d, 'L')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax1.plot(xg, mu, label = 'CV mean')
    if d_final is not None:
        f_ax1.plot(xg[:len(d_final['L'])], d_final['L'], label = 'final training')
    f_ax1.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax1.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax1.legend()
    f_ax1.set_title('Overall loss')
    # fall Risk
    f_ax2 = fig.add_subplot(gs[0, 2])
    L = arraymaker(d, 'fL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax2.plot(xg, mu)
    if d_final is not None:
        f_ax2.plot(xg[:len(d_final['fL'])], d_final['fL'])
    f_ax2.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax2.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax2.set_title('Fall risk loss')
    # msk
    f_ax3 = fig.add_subplot(gs[0, 3])
    L = arraymaker(d, 'mL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax3.plot(xg, mu)
    if d_final is not None:
        f_ax3.plot(xg[:len(d_final['mL'])], d_final['mL'])
    f_ax3.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax3.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax3.set_title('MSK prob loss')
    # nut
    f_ax4 = fig.add_subplot(gs[1, 2])
    L = arraymaker(d, 'nL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax4.plot(xg, mu)
    if d_final is not None:
        f_ax4.plot(xg[:len(d_final['nL'])], d_final['nL'])
    f_ax4.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax4.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax4.set_title('Resp Imp loss')
    # resp_imp
    f_ax5 = fig.add_subplot(gs[1, 3])
    L = arraymaker(d, 'rL')
    mu = np.nanmean(L, axis = 0)
    ciup = np.nanquantile(L, axis = 0, q = .975)
    cidn = np.nanquantile(L, axis = 0, q = .025)
    xg = list(range(len(mu)))
    f_ax5.plot(xg, mu)
    if d_final is not None:
        f_ax5.plot(xg[:len(d_final['rL'])], d_final['rL'])
    f_ax5.fill_between(xg, ciup, cidn,
                        alpha=0.2,
                         lw=2,
                         edgecolor="k")
    f_ax5.axvline(x = meanlen, linestyle = "--", color = 'black', label = "mean stop point")
    f_ax5.set_title('Nutrition loss')
    return fig


if __name__ == "__main__":
    pass