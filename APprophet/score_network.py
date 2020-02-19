# !/usr/bin/env python3

from sklearn.model_selection import KFold
from sklearn.mixture import GaussianMixture

import pandas as pd
import numpy as np


import numpy as np
import networkx as nx
from sklearn.decomposition import NMF



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


def estimate_n_clusters(combined):
    """
    Find the best number of clusters through maximization of the log-likelihood from expecation maximization
    """
    last_llh = None
    kf = KFold(n_splits=10, shuffle=True)
    components = range(50)[1:]
    X = combined.drop(['ProtA', 'ProtB']).values()
    for n_components in components:
        gm = GaussianMixture(n_components=n_components)
        llh_list = []
        for train, test in kf.split(X):
            gm.fit(X[train, :])
            if not gm.converged_:
               raise Warning("GM not converged")
            llh = -gm.score_samples(X[test, :])
            llh_list += llh.tolist()
        avg_llh = np.average(llh_list)
        print(avg_llh)
        if last_llh is None:
            last_llh = avg_llh
        elif avg_llh+10E-6 <= last_llh:
            return n_components-1
        last_llh = avg_llh
    return last_llh
