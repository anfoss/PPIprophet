# !/usr/bin/env python3

import sys
import os
import itertools
from functools import reduce
import matplotlib.pyplot as plt
import pickle

import pandas as pd
import numpy as np
import networkx as nx


from PPIprophet import io_ as io


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
        n = 0
        for G in self.networks:
            nd = list(map(str, G.nodes()))
            adj = nx.adjacency_matrix(G, nodelist=sorted(nd), weight="weight")
            self.ids = sorted(nd)
            all_adj.append(adj.todense())
            n += 1
        # now take maximum probability per element
        self.adj_matrx = np.maximum.reduce(all_adj)
        return self.adj_matrx

    def multi_collapse(self):
        self.combined = reduce(
            lambda x, y: pd.merge(x, y, on=["ProtA", "ProtB"], how="outer"), self.dfs
        )
        self.combined.fillna(0, inplace=True)
        self.combined.set_index(["ProtA", "ProtB"], inplace=True)
        self.combined["CombProb"] = np.max(self.combined.values, axis=1)
        return self.combined

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
        self.df[col] = self.df[col].str.split("_").str[0]

    def convert_to_network(self):
        # self.clean_name('ProtA')
        # self.clean_name('ProtB')
        for row in self.df.itertuples():
            self.G.add_edge(row[1], row[2], weight=row[3])
        return True

    def fill_graph(self, ids, w=10 ** -17):
        G2 = fully_connected(ids)
        [
            self.G.add_edge(*p, weight=w)
            for p in G2.edges()
            if not self.G.has_edge(p[0], p[1])
        ]
        return self.G

    def weight_adj_matrx(self, path, write=True):
        """
        creates adjacency matrix from the combined graph of all exps
        Args:
            path = outfile path
            write = wrout or not
        Returns:
            True for testing
        """
        self.adj = nx.adjacency_matrix(
            self.G, nodelist=sorted(map(str, self.G.nodes())), weight="weight"
        )
        self.adj = self.adj.todense()
        if write:
            nm = os.path.join(path, "adj_matrix.txt")
            np.savetxt(nm, self.adj, delimiter="\t")
        return True

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


def fully_connected(l, w=10 ** -17):
    G = nx.Graph()
    [G.add_edge(u, q, weight=w) for u, q in itertools.combinations(l, 2)]
    return G


def label_inte(subs):
    if subs['CombProb'] > 0.9:
        return 'High confidence'
    elif subs['CombProb'] > 0.75:
        return 'Medium confidence'
    elif subs['CombProb'] >= 0.5:
        return 'Low confidence'
    else:
        return 'No interaction'


def reshape_df(subs):
    subs = subs[subs['CombProb'] >= 0.5]
    if 'ProtB' in subs.columns:
        col = 'ProtB'
    else:
        col = 'ProtA'
    inter = list(subs[col])
    if len(inter) > 0:
        return pd.Series([",".join(inter), int(len(inter))])


def runner(tmp_, ids, outf, crapome):
    """
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    """
    if not os.path.isdir(outf):
        os.makedirs(outf)
    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    exp_info = io.read_sample_ids(ids)
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    exp_info = {strip(k): v for k, v in exp_info.items()}
    allexps = NetworkCombiner()
    allids = []
    for smpl in dir_:
        base = os.path.basename(os.path.normpath(smpl))
        if not exp_info.get(base, None):
            continue
        # print(base, exp_info[base])
        pred_out = os.path.join(smpl, "dnn.txt")
        raw_matrix = os.path.join(smpl, "transf_matrix.txt")
        allids.extend(list(pd.read_csv(raw_matrix, sep="\t")["ID"]))
        exp = TableConverter(name=exp_info[base], table=pred_out, cond=pred_out)
        exp.convert_to_network()
        exp.weight_adj_matrx(smpl, write=True)
        allexps.add_exp(exp)
    allexps.create_dfs()
    allexps.combine_graphs(allids)
    m_adj = allexps.adj_matrix_multi()
    np.savetxt(os.path.join(tmp_, "adj_mult.csv"), m_adj, delimiter=",")
    ids = allexps.get_ids()
    with open(os.path.join(tmp_, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    # protA protB format
    outname = os.path.join(outf, "adj_list.txt")
    outfile = allexps.multi_collapse()
    outfile.reset_index(inplace=True)
    outfile['confidence'] = outfile.apply(label_inte, axis=1)
    crap = io.read_crap(crapome)
    outfile['Frequency_crapome_ProtA'] = outfile['ProtA'].map(crap)
    outfile['Frequency_crapome_ProtB'] = outfile['ProtB'].map(crap)
    outfile.fillna(0, inplace=True)
    outfile.to_csv(os.path.join(outname), sep="\t", index=False)
    # adj matrix
    # m_adj = pd.DataFrame(m_adj, index=ids)
    # m_adj.columns = ids
    # m_adj.to_csv(os.path.join(outf, 'adj_matrix_combined.txt'), sep="\t")
    # TODO fix here all bool f
    mask = pd.Series.eq(outfile['ProtA'], outfile['ProtB'])
    print(outfile.shape)
    outfile = outfile[~mask]
    print(list(set(mask)))
    assert False
    reshaped = outfile.groupby(['ProtA']).apply(reshape_df).reset_index()
    reshaped.columns = ['Protein', 'interactors', '# interactors']
    reshaped2 = outfile.groupby(['ProtB']).apply(reshape_df).reset_index()
    reshaped2.columns = ['Protein', 'interactors', '# interactors']
    reshaped = pd.concat([reshaped, reshaped2]).dropna()
    outname2 = os.path.join(outf, "prot_centr.txt")
    reshaped.to_csv(os.path.join(outname2), sep="\t", index=False)
