
import multiprocessing as mp
from _99_project_module import read_pickle, write_txt
from configargparse import ArgParser
import os
import pandas as pd
import matplotlib.pyplot as plt
import re
import numpy as np


def h(x):
    """entropy"""
    return -np.sum(x * np.log(x), axis=1)

if __name__ == '__main__':

    # #########################################
    # # set some globals
    # batchstring = "02"
    # # set the seed and define the training and test sets
    # # mainseed = 8675309
    # # mainseed= 29062020 # 29 June 2020
    # mainseed = 20200813  # 13 August 2020 batch 2
    # mainseed = 20200824  # 24 August 2020 batch 2 reboot, after fixing sortedness issue
    # initialize_inprog = True
    # ##########################################
    p = ArgParser()
    p.add("--batchstring", help="the batch number", type=str)
    options = p.parse_args()
    batchstring = options.batchstring

    datadir = f"{os.getcwd()}/data/"
    outdir = f"{os.getcwd()}/output/"
    figdir = f"{os.getcwd()}/figures/"
    logdir = f"{os.getcwd()}/logs/"
    ALdir = f"{outdir}saved_models/AL{batchstring}/"

    entfiles = os.listdir(f"{ALdir}ospreds/")

    def f(i):
        r = read_pickle(f"{ALdir}ospreds/pred{i}.pkl")
        r.pop("pred")
        return r

    pool = mp.Pool(mp.cpu_count())
    edicts = pool.map(f, entfiles)
    pool.close()

    res = pd.DataFrame([i for i in edicts if i is not None])
    res.to_pickle(f"{ALdir}entropies_of_unlableled_notes.pkl")

    print('entropy summaries')

    colnames = res.columns[1:].tolist()
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(10, 10))
    for i in range(5):
        for j in range(5):
            if i == j:
                ax[i, j].hist(res[colnames[i]])
                ax[i, j].set_xlabel(colnames[i])
            elif i > j:
                ax[i, j].scatter(res[colnames[i]], [res[colnames[j]]], s=.5)
                ax[i, j].set_xlabel(colnames[i])
                ax[i, j].set_ylabel(colnames[j])
    plt.tight_layout()
    fig.savefig(f"{ALdir}entropy_summaries.pdf")
    # plt.show()

    print("entropy summaries figure")

    cndf = pd.read_pickle(f"{outdir}conc_notes_df.pkl")
    cndf = cndf.loc[cndf.LATEST_TIME < "2019-01-01"]
    cndf['month'] = cndf.LATEST_TIME.dt.month + (
            cndf.LATEST_TIME.dt.year - min(cndf.LATEST_TIME.dt.year)) * 12
    cndf_notes = ("embedded_note_m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".pkl").tolist()

    # pull the best notes
    cndf['note'] = "embedded_note_m" + cndf.month.astype(str) + "_" + cndf.PAT_ID + ".pkl"
    res = res.merge(cndf[['note', 'combined_notes']])

    best = res.sort_values("hmean_above_average", ascending=False).head(30)
    best['PAT_ID'] = best.note.apply(lambda x: x.split("_")[3][:-4])
    best['month'] = best.note.apply(lambda x: x.split("_")[2][1:])
    cndf['month'] = cndf.LATEST_TIME.dt.month

    selected_notes = []
    for i in range(len(best)):
        ni = cndf.combined_notes.loc[(cndf.month == int(best.month.iloc[i])) & (cndf.PAT_ID == best.PAT_ID.iloc[i])]
        assert len(ni) == 1
        selected_notes.append(ni)

    for i, n in enumerate(selected_notes):
        fn = f"AL{batchstring}_v2{'ALTERNATE' if i > 24 else ''}_m{best.month.iloc[i]}_{best.PAT_ID.iloc[i]}.txt"
        print(fn)
        write_txt(n.iloc[0], f"{ALdir}{fn}")

    # post-analysis
    pd.read_json(f"{ALdir}hpdf.json").to_csv(f"{ALdir}hpdf_for_R.csv")

    # get the files
    try:
        os.mkdir(f"{ALdir}best_notes_embedded")
    except Exception:
        pass

    for note in best.note:
        cmd = f"cp {ALdir}/ospreds/pred{note}.pkl" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)
    for note in best.note:
        cmd = f"cp /project/hipaa_garywlab/frailty/output/embedded_notes/{note}" \
              f" {ALdir}best_notes_embedded/"
        os.system(cmd)

    predfiles = os.listdir(f"{ALdir}best_notes_embedded")
    predfiles = [i for i in predfiles if "predembedded" in i]
    enotes = os.listdir(f"{ALdir}best_notes_embedded")
    enotes = [i for i in enotes if "predembedded" not in i]

    hpdf = pd.read_json(f"{ALdir}hpdf.json")
    winner = hpdf.loc[hpdf.best_loss == hpdf.best_loss.min()]
    best_model = pd.read_pickle(f"{ALdir}model_batch4_{int(winner.idx)}.pkl")
    out_varnames = ['MSK prob', 'Nutrition', 'Resp Imp', 'Fall Risk']

    j = 0
    for j, k in enumerate(predfiles):
        p = read_pickle(f"{ALdir}best_notes_embedded/{predfiles[j]}")
        ID = re.sub('.pkl', '', "_".join(predfiles[j].split("_")[2:]))
        emat = np.stack([h(x) for x in p['pred']]).T
        emb_note = read_pickle(f"{ALdir}best_notes_embedded/{[x for x in enotes if ID in x][0]}")
        fig, ax = plt.subplots(nrows=4, figsize=(20, 10))
        for i in range(4):
            ax[i].plot(p['pred'][i][:, 0], label='neg')
            ax[i].plot(p['pred'][i][:, 2], label='pos')
            hx = h(p['pred'][i])
            ax[i].plot(hx + 1, label='entropy')
            ax[i].legend()
            ax[i].axhline(1)
            ax[i].set_ylabel(out_varnames[i])
            ax[i].set_ylim(0, 2.1)
            maxH = np.argmax(emat[:, i])
            span = emb_note.token.iloc[(maxH - best_model['hps'][0] // 2):(maxH + best_model['hps'][0] // 2)]
            string = " ".join(span.tolist())
            ax[i].text(maxH, 2.1, string, horizontalalignment='center')
        plt.tight_layout()
        plt.show()
        fig.savefig(f"{ALdir}best_notes_embedded/predplot_w_best_span_{enotes[j]}.pdf")
