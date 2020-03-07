# !/usr/bin/env python3

import sys
import os
import pickle
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import networkx as nx

from APprophet import io_ as io
from APprophet import danmf, mcl


def normalize_wd(wd_arr, norm_=0.98):
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
        arr: 1d array array
        norm: boolean for normalization or not
    Returns:
        a single wd score for this row
    Raises:
    """
    pres = arr[arr>0].shape[0]
    npres = arr[arr==0].shape[0]
    ntot =pres+npres
    mu_ = np.sum(arr)/ntot
    sum_sq_err = np.sum((arr - mu_)**2) + ((mu_**2) * npres)
    sd_prey = np.sqrt(sum_sq_err / (ntot-1))
    wj = sd_prey / mu_
    if wj < 1:
        wj = 1
    wd_inner = (ntot / pres) * wj
    wd = arr * wd_inner
    if norm:
        return normalize_wd(wd)
    else:
        return wd


def plot_distr(wd, sim, cutoff, plotname):
    """
    plot of real and simulated data
     Args:
        wd is matrix of wd scores from calc_wd
        sim is distribution of simulated scores
     Returns:
        True
    """
    wd = wd.flatten()
    sim = sim.flatten()
    plt.figure(figsize=(6, 6))
    binNum = 100
    dist = np.unique(np.concatenate((wd, sim)))
    binwidth = (max(dist) - min(dist)) / binNum
    plt.hist(wd, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='r', edgecolor='r', alpha=0.3, label='Target')
    plt.hist(sim, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='b', edgecolor='b', alpha=0.3, label='Simulated')
    plt.ylabel('Frequency')
    plt.axvline(x=cutoff,color='green')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotname, dpi=800, bbox_inches='tight')
    plt.close()
    return True


def calc_wd_matrix(m, iteration=1000, q=0.9, norm=False, plot=False):
    """
    get a NxM matrix and calculate wd then for iteration creates dummy matrix
    and score them to get distribution of simulated interactors for each bait
    Args:
        m stats matrix following http://besra.hms.harvard.edu/ipmsmsdbs/cgi-bin/tutorial.cgi format
        iteration number of iteration for generating simulated distribution
        quantile to filter interactors for
        norm quantile based normalization of interaction
        plot boolean for plotting distribution of real and simulated data
    Returns:
        wd scores matrix
    """
    wd = np.array([vec_wd_score(m[i], norm) for i in range(m.shape[1])])
    i = 0
    rand_dist = []
    p = m.flatten()
    while i <= iteration:
        np.random.shuffle(p)
        p_arr = np.random.choice(p, m.shape[0])
        # force to have some numbers inside
        while not np.any(p_arr):
            np.random.choice(p, m.shape[0])
        # print('iteration {} of {}'.format(i, iteration))
        rand_dist.append(vec_wd_score(p_arr, norm).flatten())
        i+=1
    rand_dist = np.array(rand_dist).reshape(-1,1)
    cutoff = np.quantile(rand_dist[rand_dist > 0], q)
    if plot:
        plot_distr(wd, rand_dist, cutoff, 'test_distr.pdf')
    wd[wd < cutoff] = 0
    return wd


def rec_mcl(adj_matrix):
    """
    run mcl after converting the adjacency matrix to a matrix of weight
    """
    result = mcl.run_mcl(adj_matrix, verbose=False)
    clusters = mcl.get_clusters(result)
    opt = mcl.run_mcl(adj_matrix, inflation=optimize_mcl(adj_matrix, result, clusters), verbose=True)
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
            infl = inflation
    return infl


def output_from_clusters(nodes, clusters, out):
    idx = 1
    ids = 'complex'
    header = ["ComplexID", "ComplexName", "subunits(Gene name)"]
    path = os.path.join(out, 'communities.txt')
    io.create_file(path, header)
    for cmplx in clusters:
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
    m = m[:,~mask]
    ids = list(np.array(ids)[~mask])
    return m, ids


def to_adj_lst(adj_m):
    """
    converts adjacency matrix to adj list
    Args:
    Returns:
    Raises:
    """
    # tri_adj = np.triu(adj_m)
    # get all indexes which are valid
    idx = np.triu_indices(n=adj_m.shape[0])
    v = adj_m[idx].reshape(-1,1)
    col = idx[0].reshape(-1,1)
    row = idx[1].reshape(-1,1)
    final = np.concatenate((row, col, v), axis=1)
    return final


def heatmap(adj):
    import seaborn as sns
    import matplotlib.pyplot as plt

    adj = pd.DataFrame(adj)
    mask = np.zeros_like(adj, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True
    f, ax = plt.subplots(figsize=(11, 9))
    sns_plot = sns.clustermap(
                              adj,
                              vmax=1,
                              cmap='YlGnBu',
                              center=0.5,
                              square=True,
                              linewidths=.5,
                              cbar_kws={'shrink': .5}
                              )
    sns_plot.savefig('adj_matrix.pdf', dpi=800)


def plot_network(adj, df, community):
    import igraph as ig

    adj = pd.DataFrame(adj)
    adj_ids = list(adj.index)
    comm = max(set(list(community.values())))
    cols = ig.ClusterColoringPalette(comm + 1)
    clr = [cols.get(community[x]) for x in list(adj.index)]

    g = ig.Graph.Adjacency((adj.values > 0).tolist())
    g.to_undirected()
    weights = adj.values[np.where(adj.values)]
    g.vs['label'] = adj.index
    g.es['weight'] = weights

    # remove disconnected nodes
    g.delete_vertices(g.vs.find(_degree=0))

    ig.plot(
        g,
        'PPI_network.pdf',
        layout='fr',
        vertex_size=20,
        vertex_color=clr,
        edge_width=[x/3 for x in g.es['weight']],
        labels=True,
        vertex_frame_color=clr,
        keep_aspect_ratio=False,
    )


# @io.timeit
def runner(tmp_, outf, plots=True):
    """
    read folder tmp_ in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp_ directory
    """
    m = np.loadtxt(os.path.join(tmp_, 'adj_mult.csv'), delimiter=',')
    with open(os.path.join(tmp_, 'ids.pkl'), 'rb') as f:
        ids = pickle.load(f)
    print('calculating wd score\n')
    m, ids = preprocess_matrix(m, ids)
    # calc wd score per matrix
    wd = calc_wd_matrix(m, iteration=20, q=0.25, plot=True)
    wd_ls = to_adj_lst(wd)
    df = pd.DataFrame(wd_ls)
    ids_d = dict(zip(range(0, len(ids)), ids))
    df.columns = ['protA', 'protB', 'WD']
    df['protA'] = df['protA'].map(ids_d)
    df['protB'] = df['protB'].map(ids_d)
    df.to_csv(os.path.join(outf, 'wd_scores.txt'), sep="\t", index=False)
    # now we use wd score to filter prob
    m[wd == 0] = 0
    m,ids = preprocess_matrix(m, ids)
    print('Predicting complexes from network\n')
    clusters = rec_mcl(m)
    output_from_clusters(ids, clusters, outf)
    G = nx.from_numpy_matrix(m)
    clf = danmf.DANMF(layers=[96, 20],
                      iterations=1000,
                      pre_iterations=1000,
                      lamb=0.0001,
                      )
    clf.fit(G)
    ids_d = dict(zip(range(0, len(ids)), ids))
    out = []
    for k,v in clf.get_memberships().items():
        # this returns a dict of list where list is [cluster nr1, nr2, nr3]
        if ids_d.get(k, False):
            try:
                out.append([k, ids_d[k], ";".join(list(map(str, v)))])
            except TypeError as e:
                out.append([k, ids_d[k], str(v)])
    if plots:
        df_adj = pd.DataFrame(m, index=ids, columns=ids)
        heatmap(df_adj)
    out = pd.DataFrame(out, columns=['IDX', 'Identifier', 'Community'])
    outname = os.path.join(outf, "communities_damf.txt")
    out.to_csv(outname, sep="\t", index=False)
