# !/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.mixture import GaussianMixture

from APprophet import io_ as io
from APprophet import danmf, mcl


def plot_fdr(target, decoy, cutoff, fdr, plotname):
    """
    plot of real and simulated data
     Args:
        qvalues is matrix of qvalues scores from calc_qvalues
        sim is distribution of simulated scores
     Returns:
        True
    """
    qvalues = target.flatten()
    sim = decoy.flatten()

    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(311)
    binNum = 200
    dist = np.unique(np.concatenate((qvalues, sim)))
    binwidth = (max(dist) - min(dist)) / binNum
    plt.hist(
        qvalues,
        bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
        color="r",
        edgecolor="r",
        alpha=0.3,
        label="Target",
    )
    plt.hist(
        sim,
        bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
        color="b",
        edgecolor="b",
        alpha=0.3,
        label="Simulated",
    )
    plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
    plt.ylabel("Frequency")
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(fdr["qvalues"], fdr["target"], "r", label="P[Target>x]")
    plt.plot(fdr["qvalues"], fdr["decoy"], "b--", label="P[Decoy>x]")
    plt.plot(fdr["qvalues"], fdr["fdr"], "k", label="FDR")
    plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("qvalues score")
    plt.ylabel("Probability")
    plt.legend()
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    ax3 = plt.subplot(313)
    plt.plot(fdr["decoy"], fdr["target"], "k")
    plt.ylabel("True positive rate")
    plt.xlabel("False positive rate")
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.tight_layout()
    plt.savefig(plotname, dpi=800, bbox_inches="tight")
    plt.close()
    return True


def calc_fdr(target, decoy):
    scores = np.unique(np.concatenate((target, decoy)))
    scores = np.sort(scores)
    # initialize empty score array
    fdr = np.zeros(
        (scores.shape[0],),
        dtype=[("target", "f4"), ("decoy", "f4"), ("fdr", "f4"), ("qvalues", "f4")],
    )
    for i in range(0, scores.shape[0]):
        s = scores[i]
        nt = np.where(target >= s)[0].shape[0] / target.shape[0]
        nd = np.where(decoy >= s)[0].shape[0] / decoy.shape[0]
        if nt == 0 or (nd / nt) > 1.0:
            fdr[i, ] = (nt, nd, 1.0, s)
        else:
            fdr[i, ] = (nt, nd, nd / nt, s)
    return fdr


def calc_pdf(X):
    """
    calculate empirical fdr if not enough db hits are present
    use gaussian mixture model 2 components to predict class probability
    for all hypo and db pooled. Then from the two distributions estimated fdr
    from pep

    input == two pandas dataframe with target GO distr and decoy GO distr
    """
    X = X.reshape(-1, 1)
    # filter for > 0.5
    print(X.shape)
    X = X[X >= 0.5].reshape(-1,1)
    print(X.shape)
    clf = GaussianMixture(
        n_components=2,
        covariance_type="full",
        tol=1e-24,
        max_iter=1000,
        random_state=42,
    )
    pred_ = clf.fit(X).predict(X.reshape(-1, 1)).reshape(-1, 1)
    return np.hstack((X, pred_))


def split_posterior(X):
    """
    split classes into tp and fp based on class label after gmm fit
    """
    # force to have tp as max gmm moves label around
    d0 = X[X[:, 1] == 0][:, 0]
    d1 = X[X[:, 1] == 1][:, 0]
    if np.max(d0) > np.max(d1):
        return d0, d1
    else:
        return d1, d0


def fdr_from_pep(tp, fp, target_fdr=0.5):
    """
    estimate fdr from array generated in calc_pdf
    returns estimated fdr at each point of TP and also the go cutoff
    fdr is nr of fp > point / p > point
    """

    def fdr_point(p, fp, tp):
        fps = fp[fp >= p].shape[0]
        tps = tp[tp >= p].shape[0]
        return fps / (fps + tps)

    roll_fdr = np.vectorize(lambda p: fdr_point(p, fp, tp))
    fdr = roll_fdr(fp)
    return fdr, np.percentile(fp, target_fdr * 100)


def rescore(X, target_fdr=0.05):
    predicted = calc_pdf(X)
    tp, fp = split_posterior(predicted)
    thresh_fdr, prob_cutoff = fdr_from_pep(tp=tp, fp=fp, target_fdr=target_fdr)
    print(prob_cutoff)


def rec_mcl(adj_matrix):
    """
    run mcl after converting the adjacency matrix to a matrix of weight
    """
    result = mcl.run_mcl(adj_matrix, verbose=False)
    clusters = mcl.get_clusters(result)
    opt = mcl.run_mcl(
        adj_matrix, inflation=optimize_mcl(adj_matrix, result, clusters)
    )
    clusters = mcl.get_clusters(opt)
    return clusters


def optimize_mcl(matrix, results, clusters):
    newmax = 0
    infl = 0
    for inflation in [i / 10 for i in range(15, 26)]:
        result = mcl.run_mcl(matrix, inflation=inflation)
        clusters = mcl.get_clusters(result)
        qscore = mcl.modularity(matrix=result, clusters=clusters)
        if qscore > newmax:
            newmax = qscore
            infl = inflation
    return infl


def output_from_clusters(nodes, clusters, out):
    idx = 1
    ids = "complex"
    header = ["ComplexID", "ComplexName", "subunits(Gene name)"]
    path = os.path.join(out, "communities.txt")
    io.create_file(path, header)
    for cmplx in clusters:
        if len(list(cmplx)) > 1:
            nm = ";".join([nodes[x] for x in list(cmplx)])
            tmp = "_".join([ids, str(idx)])
            io.dump_file(path, "\t".join([str(idx), tmp, nm]))
            idx += 1
    return True


def preprocess_matrix(m, ids):
    """
    take a matrix and reshape
     Args:
     Returns:
     Raises:
    """
    # remove rows with all non significant interaction
    m[m < 0.5] = 0
    mask = np.all(m == 0, axis=1)
    m = m[~mask]
    m = m[:, ~mask]
    ids = list(np.array(ids)[~mask])
    return m, ids


def to_adj_lst(adj_m):
    """
    converts adjacency matrix to adj list
    """
    idx = np.triu_indices(n=adj_m.shape[0])
    v = adj_m[idx].reshape(-1, 1)
    col = idx[0].reshape(-1, 1)
    row = idx[1].reshape(-1, 1)
    final = np.concatenate((row, col, v), axis=1)
    return final


# @io.timeit
@io.timeit
def runner(tmp_, outf, plots=True, t=0.05):
    """
    read folder tmp_ in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp_ directory
    """
    m = np.loadtxt(os.path.join(tmp_, "adj_mult.csv"), delimiter=",")
    with open(os.path.join(tmp_, "ids.pkl"), "rb") as f:
        ids = pickle.load(f)
    # print('calculating qvalues score\n')
    m, ids = preprocess_matrix(m, ids)
    arr_qvalues = rescore(m, target_fdr=0.01)
    print(arr_qvalues)
    assert False
    qvalues_ls = to_adj_lst(arr_qvalues)
    df = pd.DataFrame(qvalues_ls)
    ids_d = dict(zip(range(0, len(ids)), ids))
    df.columns = ["protA", "protB", "qvalues"]
    df["protA"] = df["protA"].map(ids_d)
    df["protB"] = df["protB"].map(ids_d)
    df.to_csv(os.path.join(outf, "qvalue.txt"), sep="\t", index=False)
    # now we use qvalues score to filter prob
    m[arr_qvalues <= t] = 0
    m, ids = preprocess_matrix(m, ids)
    # print('Predicting complexes from network\n')
    clusters = rec_mcl(m)
    output_from_clusters(ids, clusters, outf)
    G = nx.from_numpy_matrix(m)
    clf = danmf.DANMF(
        layers=[96, 20], iterations=1000, pre_iterations=1000, lamb=0.001,
    )
    clf.fit(G)
    ids_d = dict(zip(range(0, len(ids)), ids))
    out = []
    for k, v in clf.get_memberships().items():
        # this returns a dict of list where list is [cluster nr1, nr2, nr3]
        if ids_d.get(k, False):
            try:
                out.append([k, ids_d[k], ";".join(list(map(str, v)))])
            except TypeError:
                out.append([k, ids_d[k], str(v)])
    out = pd.DataFrame(out, columns=["IDX", "Identifier", "Community"])
    outname = os.path.join(outf, "communities_damf.txt")
    out.to_csv(outname, sep="\t", index=False)
