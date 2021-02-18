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
import seaborn as sns


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
        self.combine = 'prob'

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

    def calc_freq(self, all_adj):
        """
        return a frequency matrix for positive interactions
        """

        all_adj = [np.where(x >= 0.5, 1, 0) for x in all_adj]
        # here we sum them all
        outm = np.zeros(all_adj[0].shape, dtype=np.int64)
        for x in all_adj:
            outm = outm + x
        outm = outm / len(all_adj)
        freq = np.mean(all_adj, axis=0)
        return freq

    def desi_f(self, freq, prob, w_freq=0.5, w_prob=1):
        """
        weights are hardcoded to 2 as 1:1 can be changed later
        """
        return np.exp((w_freq * np.log(freq) + w_prob * np.log(prob)) / (w_freq + w_prob))

    def adj_matrix_multi(self):
        """
        add sparse adj matrix to the adj_matrix container
        """
        all_adj = []
        n = 0
        # enforce same orderd
        self.ids = sorted(list(map(str, self.networks[0].nodes())))
        for G in self.networks:
            adj = nx.adjacency_matrix(G, nodelist=self.ids, weight="weight")
            all_adj.append(adj.todense())
            n += 1
        if self.combine == 'prob':
            # adj matrix with max prob (no filter)
            self.adj_matrx = np.maximum.reduce(all_adj)
        elif self.combine == 'combined':
            # now multiply probabilities
            self.adj_matrx = all_adj.pop()
            for m1 in all_adj:
                self.adj_matrx = np.multiply(self.adj_matrx, m1)
                # self.adj_matrx = np.add(self.adj_matrx, m1)
        elif self.combine == 'mean':
            self.adj_matrx = all_adj.pop()
            for m1 in all_adj:
                self.adj_matrx = np.add(self.adj_matrx, m1)
            self.adj_matrx / (len(all_adj) + 1)
        # calculate frequency
        # freq = self.calc_freq(all_adj)
        # self.adj_matrx = self.desi_f(freq, self.adj_matrx)
        return self.adj_matrx

    def to_adj_lst(self):
        """
        converts adjacency matrix to adj list
        """
        idx = np.triu_indices(n=self.adj_matrx.shape[0])
        v = self.adj_matrx[idx].reshape(-1, 1)
        col = idx[0].reshape(-1, 1)
        row = idx[1].reshape(-1, 1)
        final = np.concatenate((row, col, v), axis=1)
        ids_d = dict(zip(range(0, len(self.ids)), self.ids))
        df = pd.DataFrame(final)
        df.columns = ["ProtA", "ProtB", "WD"]
        df["ProtA"] = df["ProtA"].map(ids_d)
        df["ProtB"] = df["ProtB"].map(ids_d)
        #df = df[df['WD'] >= 0.5]
        print(np.max(df['WD']), np.min([df['WD']]))
        return df


    def multi_collapse(self):
        """
        this is not self.adj_matrix
        """
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

    def plot_fdr(self, target, decoy, cutoff, fdr, plotname):
        """

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
            label="Decoy",
        )
        plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
        plt.ylabel("Frequency")
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax1.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax2 = plt.subplot(312, sharex=ax1)
        plt.plot(fdr["prob"], fdr["target"], "r", label="P[Target>x]")
        plt.plot(fdr["prob"], fdr["decoy"], "b--", label="P[Decoy>x]")
        plt.plot(fdr["prob"], fdr["fdr"], "k", label="FDR")
        plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
        plt.xlabel("DNN score")
        plt.ylabel("Probability")
        plt.legend()
        box = ax2.get_position()
        ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax2.legend(loc="center left", bbox_to_anchor=(1, 0.5))

        ax3 = plt.subplot(313)
        plt.plot(fdr["decoy"], fdr["target"], "k")
        plt.plot(np.linspace(0,1, 100), np.linspace(0,1,100) , color="gray", linestyle="--", linewidth=0.5)
        plt.ylabel("True positive rate")
        plt.xlabel("False positive rate")
        box = ax3.get_position()
        ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        plt.tight_layout()
        plt.savefig(plotname, dpi=800, bbox_inches="tight")
        plt.close()
        return True

    def plot_dens(self, df, cutoff, nm):
        fig, ax = plt.subplots(figsize=(2.5, 2.5), facecolor='white')
        g = sns.FacetGrid(df, row="isdecoy", hue="isdecoy")
        g.map(sns.kdeplot,"Prob", fill=True)
        fig = g.fig
        plt.axvline(x=cutoff, color="gray", linestyle="--", linewidth=0.5)
        ax.spines["bottom"].set_color('grey')
        ax.grid(color="w", alpha=0.5)
        ax.tick_params(axis='y', which='major', labelsize=9)
        ax.tick_params(axis='x', which='minor', labelsize=6)
        ax.tick_params(axis='x', which='major', labelsize=6)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        fig.set_size_inches(5, 5, forward=True)
        fig.savefig(nm, dpi=800, bbox_inches="tight")
        plt.close()



    def fdr_control(self, q=0.5, plot=True):
        # self.df = self.df[self.df['Prob'] >= 0.5]
        decoy = self.df[self.df['isdecoy'] == 'DECOY']['Prob'].values
        target = self.df[self.df['isdecoy'] == 'TARGET']['Prob'].values
        scores = np.unique(np.concatenate((target, decoy)))
        scores = np.sort(scores)
        # initialize empty score array
        newmin = 1
        fdr = np.zeros(
            (scores.shape[0],),
            dtype=[("target", "f4"), ("decoy", "f4"), ("fdr", "f4"), ("prob", "f4")],
        )
        for i in range(0, scores.shape[0]):
            s = scores[i]
            nt = np.where(target >= s)[0].shape[0] / target.shape[0]
            nd = np.where(decoy >= s)[0].shape[0] / decoy.shape[0]
            if nt == 0 or (nd / nt) > 1.0:
                fdr[i, ] = (nt, nd, 1.0, s)
            else:
                # if the fdr is closer to q newmin is the probability
                if abs(nd/nt - q) < abs(newmin - q):
                    # print(abs(nd/nt))
                    newmin = s
                fdr[i, ] = (nt, nd, nd / nt,s)
        if plot:
            # print(newmin, self.table )

            self.plot_fdr(target, decoy, newmin, fdr, '{}prob.pdf'.format(self.table))
            self.plot_dens(self.df, newmin, '{}densplot.pdf'.format(self.table))
        self.df = self.df[self.df['Prob'] >= newmin]
        return newmin


    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df


def fully_connected(l, w=10**-17):
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
    inter = list(set(subs[col]))
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
        exp.fdr_control()
        exp.convert_to_network()
        exp.weight_adj_matrx(smpl, write=True)
        allexps.add_exp(exp)
    allexps.create_dfs()
    allexps.combine_graphs(allids)
    m_adj = allexps.adj_matrix_multi()
    # for score.py
    np.savetxt(os.path.join(tmp_, "adj_mult.csv"), m_adj, delimiter=",")
    ids = allexps.get_ids()
    with open(os.path.join(tmp_, "ids.pkl"), "wb") as f:
        pickle.dump(ids, f)

    # protA protB format
    outname = os.path.join(outf, "adj_list.txt")
    # outfile = allexps.multi_collapse()
    outfile = allexps.to_adj_lst()
    # outfile.reset_index(inplace=True)
    outfile.rename(columns={'WD': 'CombProb'}, inplace=True)
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
    #  mask = outfile['ProtA'].values == outfile['ProtB'].values
    # print(outfile)
    # outfile = outfile[~mask]
    # print(list(set(mask)))
    # assert False
    reshaped = outfile.groupby(['ProtA']).apply(reshape_df).reset_index()
    reshaped.columns = ['Protein', 'interactors', '# interactors']
    reshaped2 = outfile.groupby(['ProtB']).apply(reshape_df).reset_index()
    reshaped2.columns = ['Protein', 'interactors', '# interactors']
    reshaped = pd.concat([reshaped, reshaped2]).dropna().drop_duplicates()
    reshaped = reshaped[reshaped['Protein'] != reshaped['interactors']]
    outname2 = os.path.join(outf, "prot_centr.txt")
    reshaped.to_csv(os.path.join(outname2), sep="\t", index=False)
