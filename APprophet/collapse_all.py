# !/usr/bin/env python3

import sys
import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
import scipy
import networkx as nx
from functools import reduce

from APprophet import io_ as io



class NetworkCombiner(object):
    """
    Combine all replicates for a single condition into a network
    returns a network


    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (:obj:`int`, optional): Description of `attr2`.

    """

    def __init__(self):
        super(NetworkCombiner, self).__init__()
        self.exps = []
        self.adj_matrx = pd.DataFrame()
        self.networks = None
        self.dfs = []
        self.combined = None

    def add_exp(self, exp):
        self.exps.append(exp)

    def create_dfs(self):
        [self.dfs.append(x.get_df()) for x in self.exps]

    def add_sparse_adj(self, network):
        """
        add sparse adj matrix to the adj_matrix container
        """
        self.adj_matrx = [hstack((self.adj_matrx, X.get_adj_matrx)) for x in self.exps]
        return True

    # def matrix_factorization(R, P, Q, K, steps=5000, alpha=0.0002, beta=0.02):
    # Q = Q.T
    # for step in range(steps):
    #     for i in range(len(R)):
    #         for j in range(len(R[i])):
    #             if R[i][j] > 0:
    #                 eij = R[i][j] - np.dot(P[i,:],Q[:,j])
    #                 for k in range(K):
    #                     P[i][k] = P[i][k] + alpha * (2 * eij * Q[k][j] - beta * P[i][k])
    #                     Q[k][j] = Q[k][j] + alpha * (2 * eij * P[i][k] - beta * Q[k][j])
    #     eR = np.dot(P,Q)
    #     e = 0
    #     for i in range(len(R)):
    #         for j in range(len(R[i])):
    #             if R[i][j] > 0:
    #                 e = e + pow(R[i][j] - np.dot(P[i,:],Q[:,j]), 2)
    #                 for k in range(K):
    #                     e = e + (beta/2) * (pow(P[i][k],2) + pow(Q[k][j],2))
    #     if e < 0.001:
    #         break
    # return P, Q.T

    def multi_collapse(self, name):
        self.combined = reduce(lambda x, y: pd.merge(x, y,
                                            on = ['ProtA', 'ProtB'],
                                            how='outer'),
                    self.dfs)
        self.combined.fillna(0)
        self.combined.to_csv(name, sep="\t", index=False)



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

    def weight_adj_matrx(self, path):
        self.adj = nx.adjacency_matrix(
                                        self.G,
                                        nodelist=sorted(self.G.nodes()), weight='weight'
                                        )
        self.adj = self.adj.todense()
        nm = os.path.join(path, 'adj_matrix.txt')
        np.savetxt(nm, self.adj, delimiter="\t")
        return True

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


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
    for smpl in dir_:
        base = os.path.basename(os.path.normpath(smpl))
        if not exp_info.get(base, None):
            continue
        print(base, exp_info[base])
        pred_out = os.path.join(smpl, "dnn.txt")
        exp = TableConverter(
            name=exp_info[base],
            table=pred_out,
            cond=pred_out,
        )
        exp.convert_to_network()
        exp.weight_adj_matrx(path=smpl)
        allexps.add_exp(exp)
    allexps.create_dfs()
    outname = os.path.join(tmp_, "combined.txt")
    allexps.multi_collapse(outname)
