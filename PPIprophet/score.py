# !/usr/bin/env python3

import os
import pickle
import matplotlib.pyplot as plt
import networkx as nx
import karateclub

import pandas as pd
import numpy as np

from PPIprophet import io_ as io
from PPIprophet import mcl


def plot_fdr(target, decoy, cutoff, fdr, plotname):
    """
    plot of real and simulated data
     Args:
        wd is matrix of wd scores from calc_wd
        sim is distribution of simulated scores
     Returns:
        True
    """
    wd = target.flatten()
    sim = decoy.flatten()
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(311)
    binNum = 200
    dist = np.unique(np.concatenate((wd, sim)))
    binwidth = (max(dist) - min(dist)) / binNum
    plt.hist(
        wd,
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
    plt.plot(fdr["wd"], fdr["target"], "r", label="P[Target>x]")
    plt.plot(fdr["wd"], fdr["decoy"], "b--", label="P[Decoy>x]")
    plt.plot(fdr["wd"], fdr["fdr"], "k", label="FDR")
    plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
    plt.xlabel("WD score")
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
    # decoy = np.random.choice(decoy, target.shape[0])
    scores = np.unique(np.concatenate((target, decoy)))
    scores = np.sort(scores)
    # initialize empty score array
    fdr = np.zeros(
        (scores.shape[0],),
        dtype=[("target", "f4"), ("decoy", "f4"), ("fdr", "f4"), ("wd", "f4")],
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


def normalize_wd(wd_arr, norm_=0.9):
    """
    normalize
    Args:
        arr numpy array of wd scores
        norm_ [0,1] number corresponding to the wd_arr quantile to norm default 0.98
    Returns:
        normalized value
    """
    return wd_arr / np.quantile(wd_arr[wd_arr > 0], norm_)


def vec_wd_score(arr, norm):
    """
    vectorized wd score
    Args:
        arr: 1d array
        norm: boolean for normalization or not
    Returns:
        a single wd score for this row
    Raises:
    """
    # need to elevate to the distance from the bait
    pres = arr[arr > 0].shape[0]
    npres = arr[arr == 0].shape[0]
    if pres == 0:
        return np.zeros(arr.shape)
    ntot = pres + npres
    mu_ = np.sum(arr) / ntot
    sum_sq_err = np.sum((arr - mu_) ** 2) + ((mu_ ** 2) * npres)
    sd_prey = np.sqrt(sum_sq_err / (ntot - 1))
    wj = sd_prey / mu_
    if wj < 1:
        wj = 1
    wd_inner = (ntot / pres) * wj
    wd = arr * wd_inner
    if norm:
        return normalize_wd(wd)
    else:
        return wd


def calc_wd_matrix(m, iteration=1000, q=0.9, norm=False, plot=False):
    """
    get a NxM matrix and calculate wd then for iteration creates dummy matrix
    and score them to get distribution of simulated interactors for each bait
    Args:
        -m: http://besra.hms.harvard.edu/ipmsmsdbs/cgi-bin/tutorial.cgi format
        iteration number of iteration for generating simulated distribution
        -q:  quantile to filter interactors for
        -norm: quantile based normalization of interaction
        plot boolean for plotting distribution of real and simulated data
    Returns:
        wd: scores matrix filtered
    """
    # convert to information content
    # convert to entropy
    m = np.array([-np.log2(1 - m[i]) for i in range(m.shape[1])])
    wd = np.array([vec_wd_score(m[i], norm) for i in range(m.shape[1])])
    i = 0
    rand_dist = []
    while i <= iteration:
        p_arr = np.random.lognormal(size=m.shape[0])
        # print('iteration {} of {}'.format(i, iteration))
        rand_dist.append(vec_wd_score(p_arr, norm).flatten())
        i += 1
    rand_dist = np.array(rand_dist).flatten()
    cutoff = np.quantile(rand_dist, q)
    if plot:
        fdr = calc_fdr(wd.flatten(), rand_dist)
        plot_fdr(wd, rand_dist, cutoff, fdr, "test_distr.pdf")
    wd[wd < cutoff] = 0
    return wd


def rec_mcl(adj_matrix):
    """
    run mcl after converting the adjacency matrix to a matrix of weight
    """
    result = mcl.run_mcl(adj_matrix, verbose=False)
    clusters = mcl.get_clusters(result)
    opt = mcl.run_mcl(
                      adj_matrix,
                      expansion=2,
                      inflation=optimize_mcl(adj_matrix, result, clusters)
                      )
    clusters = mcl.get_clusters(opt)
    return clusters


def optimize_mcl(matrix, results, clusters):
    newmax = 0
    infl = 2
    for inflation in np.linspace(1.1, 5, 500):
        result = mcl.run_mcl(matrix, inflation=inflation)
        clusters = mcl.get_clusters(result)
        qscore = mcl.modularity(matrix=result, clusters=clusters)
        if qscore > newmax:
            # print('Updating Q={} from {}'.format(qscore, newmax))
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


def preprocess_matrix(m, ids, cutoff=0.5):
    """
    take a matrix and reshape
     Args:
     Returns:
     Raises:
    """
    # remove rows with all non significant interaction
    m[m < cutoff] = 0
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


def filter_crap(m, ids, crap, thres):
    """
    read crapome and applies frequency filter. need to work on GENE NAMES
    Args:
    Returns:
    Raises:
    """
    def freq(x):
        return x[x > 0].shape[0] / x.shape[0]
    crap = pd.read_csv(crap, sep="\t")
    crap.set_index('Gene', inplace=True)
    crap.drop(['RefSeq', 'UniProt'], axis=1, inplace=True)
    crap['ss'] = crap.apply(lambda x: freq(x.values), axis=1)
    # print(crap[crap.index.str.contains('^POL')])
    crap = crap[crap['ss'] >= float(thres)]
    mask = np.isin(np.array(ids), crap.index.values)
    ids = list(np.array(ids)[~mask])
    m = m[~mask]
    m = m[:, ~mask]
    return m, ids


@io.timeit
def runner(tmp_, outf, crapome, thresh):
    """
    read folder tmp_ in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp_ directory
    """
    m = np.loadtxt(os.path.join(tmp_, "adj_mult.csv"), delimiter=",")
    with open(os.path.join(tmp_, "ids.pkl"), "rb") as f:
        ids = pickle.load(f)
    # print('calculating wd score\n')
    m, ids = filter_crap(m, ids, crapome, thresh)
    m, ids = preprocess_matrix(m, ids)
    wd = calc_wd_matrix(m, iteration=10000, q=0.30, norm=False, plot=False)
    wd_ls = to_adj_lst(wd)
    df = pd.DataFrame(wd_ls)
    ids_d = dict(zip(range(0, len(ids)), ids))
    df.columns = ["ProtA", "ProtB", "WD"]
    df["ProtA"] = df["ProtA"].map(ids_d)
    df["ProtB"] = df["ProtB"].map(ids_d)
    df.to_csv(os.path.join(outf, "d_scores.txt"), sep="\t", index=False)
    # now we need to filter m
    m[wd==0] = 0
    m, ids = preprocess_matrix(m, ids)
    clusters = rec_mcl(m)
    output_from_clusters(ids, clusters, outf)
    spl = karateclub.EgoNetSplitter(10)
    spl.fit(nx.from_numpy_matrix(m))
    out = {}
    ids_d = dict(zip(range(0, len(ids)), ids))
    for k, v in spl.get_memberships().items():
        for cl_id in v:
            if out.get(cl_id):
                out[cl_id].append(ids_d[k])
            else:
                out[cl_id] = [ids_d[k]]
    todf = {k: ','.join(v) for k, v in out.items()}
    out = pd.DataFrame.from_dict(todf, orient='index')
    outname = os.path.join(outf, "communities_out.txt")
    out.to_csv(outname, sep="\t", index=True)
