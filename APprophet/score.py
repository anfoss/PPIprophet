# !/usr/bin/env python3

import sys
import os
import itertools
from functools import reduce
import matplotlib.pyplot as plt


import pandas as pd
import numpy as np
import networkx as nx


from APprophet import io_ as io
from APprophet import danmf, bigclam, edmot, ego_splitter, nnsed, scd, mnmf, symmnmf


class NetworkCombiner(object):
    """
    Combine all replicates for a single condition into a network
    returns a network
    """
    def __init__(self):
        super(NetworkCombiner, self).__init__()
        self.exps = []
        self.adj_matrx = None
        self.networks = None
        self.dfs = []
        self.ids = None
        self.combined = None

    def add_exp(self, exp):
        self.exps.append(exp)

    def create_dfs(self):
        [self.dfs.append(x.get_df()) for x in self.exps]

    def combine_graphs(self, ids_all):
        """
        fill graphs with all proteins missing and weight 0 (i.e no connection)
        """
        self.networks = [x.fill_graph(ids_all) for x in self.exps]
        return True

    def adj_matrix_multi(self):
        """
        add sparse adj matrix to the adj_matrix container
        """
        all_adj = []
        for G in self.networks:
            nd = list(map(str, G.nodes()))
            adj = nx.adjacency_matrix(
                                    G,
                                    nodelist=sorted(nd),
                                    weight='weight'
                                    )
            self.ids = sorted(nd)
            all_adj.append(adj.todense())
        # now multiply
        self.adj_matrx = all_adj.pop()
        for m1 in all_adj:
            self.adj_matrx = np.matmul(self.adj_matrx, m1)
        self.adj_matrx[self.adj_matrx < 0.5] = 0
        return self.adj_matrx

    def multi_collapse(self):
        self.combined = reduce(lambda x, y: pd.merge(x, y,
                                            on = ['ProtA', 'ProtB'],
                                            how='outer'),
                    self.dfs)
        self.combined.fillna(0, inplace=True)
        return self.combined
        # self.combined.to_csv(name, sep="\t", index=False)

    def get_ids(self):
        return self.ids

    def get_adj(self):
        return self.adj_matrx


class TableConverter(object):
    """docstring for TableConverter"""
    def __init__(self, name, table, cond):
        super(TableConverter, self).__init__()
        self.name = name
        self.table = table
        self.df = pd.read_csv(table, sep="\t")
        self.cond = cond
        self.G = nx.Graph()
        self.adj = None
        self.fdr = None

    def clean_name(self, col):
        self.df[col] = self.df[col].str.split('_').str[0]

    def convert_to_network(self):
        self.clean_name('ProtA')
        self.clean_name('ProtB')
        for row in self.df.itertuples():
            self.G.add_edge(row[1], row[2], weight=row[3])
        return True

    def fill_graph(self, ids, w=10**-17):
        G2 = fully_connected(ids)
        [self.G.add_edge(*p, weight=w) for p in G2.edges() if not self.G.has_edge(p[0], p[1])]
        return self.G

    def weight_adj_matrx(self, path, write=False):
        self.adj = nx.adjacency_matrix(
                                        self.G,
                                        nodelist=sorted(self.G.nodes()), weight='weight'
                                        )
        self.adj = self.adj.todense()
        if write:
            nm = os.path.join(path, 'adj_matrix.txt')
            np.savetxt(nm, self.adj, delimiter="\t")
        return True

    def modify_df(self, fdr_thresh):
        """
        substitute 0 to everything below fdr threshold
        """
        self.df['Prob'].values[self.df['Prob'] <= fdr_thresh] = 0

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df

    def calc_fdr(self, path, bait='UXT', target_fdr=0.3):
        """
        get shell level of interaction of bait and then calc local fdr
        for every pred level
        """
        bait_int = self.df[(self.df['ProtA']==bait) | (self.df['ProtB']==bait)]
        bait_int = bait_int[bait_int['Prob']>=0.5]
        allz = list(bait_int['ProtA'])
        allz.extend(list(bait_int['ProtB']))
        allz = set(allz)

        pos = [bait_int]
        for p in allz:
            tmp = self.df[(self.df['ProtA']==p) | (self.df['ProtB']==p)]
            pos.append(tmp[tmp['Prob'] >=0.5])
        pos = pd.concat(pos)
        neg = pd.concat([self.df,pos]).drop_duplicates(keep=False)
        pos, neg = pos['Prob'].values, neg['Prob'].values
        scores = np.concatenate((pos, neg))
        scores = np.sort(scores)
        # initialize empty score array
        fdr = np.zeros((scores.shape[0],),
                       dtype=[('target', 'f4'), ('decoy', 'f4'), ('fdr', 'f4'), ('prob', 'f4')])
        for i in range(0, scores.shape[0]):
            s = scores[i]
            nt = np.where(pos >= s)[0].shape[0] / pos.shape[0]
            nd = np.where(neg >= s)[0].shape[0] / neg.shape[0]
            if nt == 0 or (nd / nt) > 1.0:
                fdr[i, ] = (nt, nd, 1.0, s)
            else:
                fdr[i, ] = (nt, nd, nd / nt, s)
        plot_fdr(pos, neg, fdr, path)
        fdr_df = pd.DataFrame(fdr, columns=['target', 'decoy', 'fdr', 'prob'])
        fdr_df.to_csv(os.path.join(path, 'fdr.txt'), index=False, sep="\t")
        fdr_thresh = fdr_df[fdr_df['fdr'] <= target_fdr]['prob'].values[0]
        self.modify_df(fdr_thresh)


def plot_fdr(target_dist, decoy_dist, fdr, path):
    plt.figure(figsize=(6, 6))
    plt.subplot(2, 1, 1)
    binNum = 100.0
    dist = np.unique(np.concatenate((target_dist, decoy_dist)))
    binwidth = (max(dist) - min(dist)) / binNum

    plt.hist(target_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='r', edgecolor='r', alpha=0.3, label='Target')
    plt.hist(decoy_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='b', edgecolor='b', alpha=0.3, label='Decoy')
    # plt.xlabel('DNN probability')
    plt.ylabel('Frequency')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(fdr['prob'], fdr['target'], 'r', label='P[Target>x]')
    plt.plot(fdr['prob'], fdr['decoy'], 'b--', label='P[Decoy>x]')
    plt.plot(fdr['prob'], fdr['fdr'], 'k', label='FDR')
    plt.xlabel('DNN probability')
    plt.legend()

    # plt.subplot(3, 1, 3)
    # plt.plot(fdr['decoy'], fdr['target'], 'k')
    # plt.ylabel('True positive rate')
    # plt.xlabel('False positive rate')

    plt.tight_layout()
    outfile = os.path.join(path, 'fdr.pdf')
    plt.savefig(outfile, dpi=800, bbox_inches='tight')
    return True

def fully_connected(l, w=10**-17):
    G = nx.Graph()
    [G.add_edge(u,q, weight=w) for u,q in itertools.combinations(l,2)]
    return G


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
                              cmap="YlGnBu",
                              center=0.5,
                              square=True,
                              linewidths=.5,
                              cbar_kws={"shrink": .5}
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

    # edge_s = [x/3  for x in list(df['Prob'])]

    weights = adj.values[np.where(adj.values)]
    g.vs['label'] = adj.index
    g.es["weight"] = weights

    ig.plot(
        g,
        'PPI_network.pdf',
        layout="fr",
        vertex_size=20,
        vertex_color=clr,
        edge_width=[x/3 for x in g.es["weight"]],
        labels=True,
        vertex_frame_color=clr,
        keep_aspect_ratio=False,
    )


@io.timeit
def runner(tmp_, ids):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    exp_info = io.read_sample_ids(ids)
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    exp_info = {strip(k): v for k, v in exp_info.items()}
    wrout = []
    allexps = NetworkCombiner()
    allids = []
    for smpl in dir_:
        base = os.path.basename(os.path.normpath(smpl))
        if not exp_info.get(base, None):
            continue
        print(base, exp_info[base])
        pred_out = os.path.join(smpl, "dnn.txt")
        raw_matrix = os.path.join(smpl, "transf_matrix.txt")
        allids.extend(list(pd.read_csv(raw_matrix, sep="\t")['ID']))
        exp = TableConverter(
            name=exp_info[base],
            table=pred_out,
            cond=pred_out
        )
        exp.convert_to_network()
        exp.calc_fdr(smpl)
        exp.convert_to_network()
        allexps.add_exp(exp)
    allexps.create_dfs()
    allexps.combine_graphs(allids)
    m_adj = allexps.adj_matrix_multi()
    ids = allexps.get_ids()
    G = nx.from_numpy_matrix(np.array(m_adj))
    print('Predicting complexes from network\n')
    # test
    clf = edmot.EdMot()
    clf.fit(G)
    ids_d = dict(zip(range(0, len(ids)), ids))
    out = []
    for k,v in clf.get_memberships().items():
        out.append([k,ids_d.get(k, None), v])
    print(len(set(list(clf.get_memberships().values()))))

    # communities
    out = pd.DataFrame(out, columns=['IDX', 'Identifier', 'Community'])
    outname = os.path.join(tmp_, "communities.txt")
    out.to_csv(outname, sep="\t", index=False)

    # protA protB format
    outname = os.path.join(tmp_, "combined.txt")
    outfile = allexps.multi_collapse()
    outfile.to_csv(outname, sep="\t", index=False)

    # adj matrix
    m_adj = pd.DataFrame(m_adj, index=ids)
    m_adj.columns = ids
    m_adj.to_csv('adj_matrix_combined.txt', sep="\t")
    gn2comm = {ids_d.get(k, None):v for k,v in clf.get_memberships().items()}
    plot_network(m_adj, outfile, gn2comm)
