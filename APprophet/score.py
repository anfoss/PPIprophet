# !/usr/bin/env python3

import sys
import os
import itertools
from functools import reduce


import pandas as pd
import numpy as np
import networkx as nx
from sklearn.decomposition import NMF
from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

from APprophet import io_ as io


class Estimator(object):
    """Estimator base class with constructor and public methods."""

    def __init__(self):
        """Creatinng an estimator."""
        pass

    def fit(self):
        """Fitting a model."""
        pass

    def get_embedding(self):
        """Getting the embeddings (graph or node level)."""
        return None

    def get_memberships(self):
        """Getting the membership dictionary."""
        return None

    def get_cluster_centers(self):
        """Getting the cluster centers."""
        return None



class DANMF(Estimator):
    r"""An implementation of `"DANMF" <https://smartyfh.com/Documents/18DANMF.pdf>`_
    from the CIKM '18 paper "Deep Autoencoder-like Nonnegative Matrix Factorization for
    Community Detection". The procedure uses telescopic non-negative matrix factorization
    in order to learn a cluster membership distribution over nodes. The method can be
    used in an overlapping and non-overlapping way.
    Args:
        layers (list): Autoencoder layer sizes in a list of integers. Default [32, 8].
        pre_iterations (int): Number of pre-training epochs. Default 100.
        iterations (int): Number of training epochs. Default 100.
        seed (int): Random seed for weight initializations. Default 42.
        lamb (float): Regularization parameter. Default 0.01.
    """
    def __init__(self, layers=[32, 8], pre_iterations=100, iterations=100, seed=42, lamb=0.01):
        self.layers = layers
        self.pre_iterations = pre_iterations
        self.iterations = iterations
        self.seed = seed
        self.lamb = lamb
        self.p = len(self.layers)


    def _setup_target_matrices(self, graph):
        """
        Setup target matrix for pre-training process.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph being clustered.
        """
        self.graph = graph
        self.A = nx.adjacency_matrix(self.graph, nodelist=range(self.graph.number_of_nodes()))
        self.L = nx.laplacian_matrix(self.graph, nodelist=range(self.graph.number_of_nodes()))
        self.D = self.L+self.A

    def _setup_z(self, i):
        """
        Setup target matrix for pre-training process.
        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            self.Z = self.A
        else:
            self.Z = self.V_s[i-1]

    def _sklearn_pretrain(self, i):
        """
        Pre-training a single layer of the model with sklearn.
        Arg types:
            * **i** *(int)* - The layer index.
        """
        nmf_model = NMF(n_components=self.layers[i],
                        init="random",
                        random_state=self.seed,
                        max_iter=self.pre_iterations)

        U = nmf_model.fit_transform(self.Z)
        V = nmf_model.components_
        return U, V

    def _pre_training(self):
        """
        Pre-training each NMF layer.
        """
        self.U_s = []
        self.V_s = []
        for i in range(self.p):
            self._setup_z(i)
            U, V = self._sklearn_pretrain(i)
            self.U_s.append(U)
            self.V_s.append(V)

    def _setup_Q(self):
        """
        Setting up Q matrices.
        """
        self.Q_s = [None for _ in range(self.p+1)]
        self.Q_s[self.p] = np.eye(self.layers[self.p-1])
        for i in range(self.p-1, -1, -1):
            self.Q_s[i] = np.dot(self.U_s[i], self.Q_s[i+1])

    def _update_U(self, i):
        """
        Updating left hand factors.
        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            R = self.U_s[0].dot(self.Q_s[1].dot(self.VpVpT).dot(self.Q_s[1].T))
            R = R+self.A_sq.dot(self.U_s[0].dot(self.Q_s[1].dot(self.Q_s[1].T)))
            Ru = 2*self.A.dot(self.V_s[self.p-1].T.dot(self.Q_s[1].T))
            self.U_s[0] = (self.U_s[0]*Ru)/np.maximum(R, 10**-10)
        else:
            R = self.P.T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.VpVpT).dot(self.Q_s[i+1].T)
            R = R+self.A_sq.dot(self.P).T.dot(self.P).dot(self.U_s[i]).dot(self.Q_s[i+1]).dot(self.Q_s[i+1].T)
            Ru = 2*self.A.dot(self.P).T.dot(self.V_s[self.p-1].T).dot(self.Q_s[i+1].T)
            self.U_s[i] = (self.U_s[i]*Ru)/np.maximum(R, 10**-10)

    def _update_P(self, i):
        """
        Setting up P matrices.
        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i == 0:
            self.P = self.U_s[0]
        else:
            self.P = self.P.dot(self.U_s[i])

    def _update_V(self, i):
        """
        Updating right hand factors.
        Arg types:
            * **i** *(int)* - The layer index.
        """
        if i < self.p-1:
            Vu = 2*self.A.dot(self.P).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])+self.V_s[i]
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)
        else:
            Vu = 2*self.A.dot(self.P).T+(self.lamb*self.A.dot(self.V_s[i].T)).T
            Vd = self.P.T.dot(self.P).dot(self.V_s[i])
            Vd = Vd + self.V_s[i]+(self.lamb*self.D.dot(self.V_s[i].T)).T
            self.V_s[i] = self.V_s[i] * Vu/np.maximum(Vd, 10**-10)

    def _setup_VpVpT(self):
        self.VpVpT = self.V_s[self.p-1].dot(self.V_s[self.p-1].T)

    def _setup_Asq(self):
        self.A_sq = self.A.dot(self.A.T)

    def get_embedding(self):
        r"""Getting the bottleneck layer embedding.
        Return types:
            * **embedding** *(Numpy array)* - The bottleneck layer embedding of nodes.
        """
        embedding = [self.P, self.V_s[-1].T]
        embedding = np.concatenate(embedding, axis=1)
        return embedding

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.
        Return types:
            * **memberships** *(dict)*: Node cluster memberships.
        """
        index = np.argmax(self.P, axis=1)
        memberships = {int(i): int(index[i]) for i in range(len(index))}
        return memberships

    def fit(self, graph):
        """
        Fitting a DANMF clustering model.
        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self._setup_target_matrices(graph)
        self._pre_training()
        self._setup_Asq()
        for iteration in range(self.iterations):
            self._setup_Q()
            self._setup_VpVpT()
            for i in range(self.p):
                self._update_U(i)
                self._update_P(i)
                self._update_V(i)


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

    def combine_graphs(self, l):
        """
        fill graphs with all proteins missing and weight 0 (i.e no connection)
        """
        ids_all = list(set([x.split('_')[0] for x in l]))
        # fill graph
        self.networks = [x.fill_graph(ids_all) for x in self.exps]
        return True

    def adj_matrix_multi(self):
        """
        add sparse adj matrix to the adj_matrix container
        """
        all_adj = []
        for G in self.networks:
            adj = nx.adjacency_matrix(
                                    G,
                                    nodelist=sorted(G.nodes()),
                                    weight='weight'
                                    )
            self.ids = sorted(G.nodes())
            all_adj.append(adj.todense())
        # now multiply
        self.adj_matrx = all_adj.pop()
        for m1 in all_adj:
            self.adj_matrx = np.matmul(self.adj_matrx, m1)
        # self.adj_matrx[self.adj_matrx < 0.5] = 0
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

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


def fully_connected(l, w=10**-17):
    # add small values as weight to get value in matrix multiplication
    G = nx.Graph()
    [G.add_edge(u,q, weight=w) for u,q in itertools.combinations(l,2)]
    return G


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
        # create base network
        exp.convert_to_network()
        # exp.weight_adj_matrx(path=smpl)
        allexps.add_exp(exp)
    allexps.create_dfs()
    # combine all individual graphs
    allexps.combine_graphs(allids)
    # extract combined adjancency matrix for all samples
    m_adj = allexps.adj_matrix_multi()
    ids = allexps.get_ids()
    G = nx.from_numpy_matrix(np.array(m_adj))
    print('Predicting complexes from network\n')
    # test
    clf = DANMF(
                layers=[500, 100],
                pre_iterations=50,
                iterations=200,
                seed=42,
                lamb=0.01)
    clf.fit(G)
    ids = dict(zip(range(0, len(ids)), ids))
    outname = os.path.join(tmp_, "combined.txt")
    outfile = allexps.multi_collapse()
    outfile.to_csv(outname, sep="\t", index=False)
    out = []
    for k,v in clf.get_memberships().items():
        out.append([k,ids[k], v])
    print(len(set(list(clf.get_memberships().values()))))
    out = pd.DataFrame(out, columns=['IDX', 'Identifier', 'Community'])
    outname = os.path.join(tmp_, "communities.txt")
    out.to_csv(outname, sep="\t", index=False)
