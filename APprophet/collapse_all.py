# !/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import scipy
import networkx as nx
from functools import reduce
import itertools

from APprophet import io_ as io



class NetworkCombiner(object):
    """
    Combine all replicates for a single condition into a network
    returns a network
    """
    def __init__(self):
        super(NetworkCombiner, self).__init__()
        self.exps = []
        self.adj_matrx = pd.DataFrame()
        self.networks = None
        self.dfs = []
        self.ids = None
        self.combined = None

    def add_exp(self, exp):
        self.exps.append(exp)

    def create_dfs(self):
        [self.dfs.append(x.get_df()) for x in self.exps]

    def combine_graphs(self, l):
        """
        fill graphs with all proteins missing and weight 0 (i.e no connection)
        """
        ids_all = list(set(l))
        # fill graph
        self.networks = [x.fill_graph(ids_all) for x in self.exps]
        self.ids = ids_all
        return True

    def adj_matrix_multi(self, network):
        """
        add sparse adj matrix to the adj_matrix container
        """
        adj = []
        for G in self.networks:
            self.adj = nx.adjacency_matrix(
                                            G,
                                            nodelist=sorted(G.nodes()), weight='weight'
                                            )
            self.adj = self.adj.todense()
            print(self.adj.shape)
        # now sum all adj matrixes to keep the max
        return True

    def multi_collapse(self, name):
        self.combined = reduce(lambda x, y: pd.merge(x, y,
                                            on = ['ProtA', 'ProtB'],
                                            how='outer'),
                    self.dfs)
        self.combined.fillna(0)
        self.combined.to_csv(name, sep="\t", index=False)

    def predict_membership(self):
        """
        performs overlapping community dectection from the factorize_adj_matrix
        uses # XXX:
        """
        pass

    def factorize_adj_matrix(self):
        """
        """
        pass


class TableConverter(object):
    """docstring for TableConverter"""
    def __init__(self, name, table, cond):
        super(TableConverter, self).__init__()
        self.name = name
        self.table = table
        self.df = None
        self.cond = cond
        self.G = nx.Graph()
        self.adj = None

    def clean_name(self, col):
        self.df[col] = self.df[col].str.split('_').str[0]

    def convert_to_network(self):
        self.df = pd.read_csv(self.table, sep="\t")
        self.clean_name('ProtA')
        self.clean_name('ProtB')
        for row in self.df.itertuples():
            self.G.add_edge(row[1], row[2], weight=row[3])
        return True

    def fill_graph(self, ids):
        G2 = fully_connected(ids)
        [self.G.add_edge(*p, weight=0) for p in G2.edges() if not self.G.has_edge(p[0], p[1])]
        return True

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

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


def fully_connected(l, create_using=None):
    G = nx.Graph()
    [G.add_edge(u,q, weight=0) for u,q in itertools.combinations(l,2)]
    return G


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
        # raw_matrix = os.path.join(smpl, "transf_matrix.txt")
        allids.extend(list(pd.read_csv(raw_matrix, sep="\t")['ID']))
        exp = TableConverter(
            name=exp_info[base],
            table=pred_out,
            cond=pred_out
        )
        # create base network
        exp.convert_to_network()
        # exp.weight_adj_matrx(path=smpl)
        allexps.add_exp(exp)
    allexps.create_dfs()
    # combine all individual graphs
    allexps.combine_graphs(allids)
    # extract combined adjancency matrix for all samples
    allexps.adj_matrix_multi()
    outname = os.path.join(tmp_, "combined.txt")
    allexps.multi_collapse(outname)
