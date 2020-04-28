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


from APprophet import io_ as io


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
        # now multiply probabilities
        self.adj_matrx = all_adj.pop()
        for m1 in all_adj:
            self.adj_matrx = np.multiply(self.adj_matrx, m1)
            # self.adj_matrx = np.add(self.adj_matrx, m1)
        # return self.adj_matrx / (len(all_adj) + 1)
        return self.adj_matrx

    def multi_collapse(self):
        self.combined = reduce(
            lambda x, y: pd.merge(x, y, on=["ProtA", "ProtB"], how="outer"), self.dfs
        )
        self.combined.fillna(0, inplace=True)
        self.combined.set_index(["ProtA", "ProtB"], inplace=True)
        self.combined["CombProb"] = np.prod(self.combined.values, axis=1)
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

    # def modify_weight(self, bait):
    #     """
    #     modify bait with sqrt weight if bait not interacting
    #     Args:
    #     Returns:
    #     Raises:
    #     """
    #     for e1, e2 in self.G.edges():
    #         if e1 != bait and e2 != bait:
    #             w = self.G[e1][e2]['weight']
    #             self.G[e1][e2]['weight'] = w**0.5
    #         elif self.G[e1][e2]['weight'] < 0.5:
    #             w = self.G[e1][e2]['weight']
    #             self.G[e1][e2]['weight'] = w**0.5
    #         else:
    #             pass

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


def fully_connected(l, w=10 ** -17):
    G = nx.Graph()
    [G.add_edge(u, q, weight=w) for u, q in itertools.combinations(l, 2)]
    return G


def runner(tmp_, ids, outf):
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
    wrout = []
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
        # exp.modify_weight(bait='UXT')
        exp.weight_adj_matrx(smpl, write=True)
        allexps.add_exp(exp)
    allexps.create_dfs()
    allexps.combine_graphs(allids)
    m_adj = allexps.adj_matrix_multi()
    ids = allexps.get_ids()
    with open(os.path.join(tmp_, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    # protA protB format
    outname = os.path.join(outf, "adj_list.txt")
    outfile = allexps.multi_collapse()
    outfile.to_csv(os.path.join(outname), sep="\t", index=True)

    # adj matrix
    np.savetxt(os.path.join(tmp_, "adj_mult.csv"), m_adj, delimiter=",")
    # m_adj = pd.DataFrame(m_adj, index=ids)
    # m_adj.columns = ids
    # m_adj.to_csv(os.path.join(outf, 'adj_matrix_combined.txt'), sep="\t")
