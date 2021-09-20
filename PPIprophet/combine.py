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
from functools import reduce


from PPIprophet import io_ as io
from PPIprophet import qvalue as qvalue




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
        # used mean for paper
        self.combine = 'prob'
        self.comb_net = None

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

    def multi_prob_out(self):
        """
        print output in the format of prota and protb with probabilities per file
        """
        allprobs = []
        for G in self.networks:
            # adj = nx.adjacency_matrix(G, nodelist=self.ids, weight="weight")
            df = nx.to_pandas_edgelist(G)
            df2 = df.copy()
            df2.columns = ['target', 'source', 'weight']
            df2 = df2[['source', 'target', 'weight']]
            df = pd.concat([df, df2])
            allprobs.append(df)
        tots = reduce(lambda l,r: pd.merge(l,r,on=['source', 'target'], how='outer'), allprobs)
        # replace interactions with all NAN to 0
        tots.set_index(['source','target'])
        tots = tots.replace(10**-17, np.nan)
        tots.dropna(how='all', axis=0, inplace=True)
        return tots

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
        need to be done per group
        """
        all_adj = []
        # enforce same order
        self.ids = sorted(list(map(str, self.networks[0].nodes())))
        for G in self.networks:
            adj = nx.adjacency_matrix(G, nodelist=self.ids, weight="weight")
            all_adj.append(adj.todense())
        if self.combine == 'prob':
            # adj matrix with max prob (no filter)
            self.adj_matrx = np.maximum.reduce(all_adj)
        elif self.combine == 'comb':
            # now multiply probabilities
            self.adj_matrx = all_adj.pop()
            for m1 in all_adj:
                self.adj_matrx = np.multiply(self.adj_matrx, m1)
                # self.adj_matrx = np.add(self.adj_matrx, m1)
        elif self.combine == 'mean':
            self.adj_matrx = all_adj.pop()
            for m1 in all_adj:
                self.adj_matrx = np.add(self.adj_matrx, m1)
            self.adj_matrx = self.adj_matrx / (len(all_adj) + 1)
        # calculate frequency
        # freq = self.calc_freq(all_adj)
        # self.adj_matrx = self.desi_f(freq, self.adj_matrx)
        # convert to network and return
        # nodes are alphabetically ordered
        self.comb_net = nx.from_numpy_matrix(self.adj_matrx)
        self.comb_net = nx.relabel_nodes(self.comb_net, dict(zip(self.comb_net.nodes, self.ids), copy=False))
        return self.adj_matrx

    def to_adj_lst(self):
        """
        converts adjacency matrix to adj list
        """
        df = nx.to_pandas_edgelist(self.comb_net)
        df.columns = ["ProtA", "ProtB", "CombProb"]
        df = df[df['ProtA'] != df['ProtB']]
        df = df[df['CombProb'] >= 0.5]
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

    def get_gr_network(self):
        """
        convert self.adj_matrx to network using node ids
        """
        G = nx.from_numpy_matrix(self.adj_matrx)
        G = nx.relabel_nodes(G,dict(zip(G.nodes, self.ids)))
        return G



class TableConverter(object):
    """docstring for TableConverter"""

    def __init__(self, table, cond):
        super(TableConverter, self).__init__()
        self.table = table
        self.df = pd.read_csv(table, sep="\t")
        self.cond = cond
        self.G = nx.Graph()
        self.adj = None
        # used 0.75 in paper
        self.cutoff_fdr = 1

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

    # used 0.75 for paper and no split in preprocess
    def fdr_control(self, plot=True):
        self.df = self.df[self.df['Prob'] >= 0.5]
        if self.cutoff_fdr < 1 :
            decoy = self.df[self.df['isdecoy'] == 'DECOY']['Prob'].values
            target = self.df[self.df['isdecoy'] == 'TARGET']['Prob'].values
            error_stat = qvalue.error_statistics(target, decoy)
            i0 = (error_stat.qvalue - self.cutoff_fdr).abs().idxmin()
            self.cutoff_fdr = error_stat.iloc[i0]["cutoff"]
            print("cutoff for {} is {}".format(self.table, self.cutoff_fdr))
            # self.df = self.df[self.df['Prob'] >= self.cutoff_fdr]
            # 0s the probability below fdr thresholds
            self.df[self.df['Prob'] <= self.cutoff_fdr] = 10**-17
            self.df = self.df[self.df['isdecoy'] == 'TARGET']
            pth = os.path.dirname(os.path.abspath(self.table))
            error_stat.to_csv('{}/error_metrics.txt'.format(pth), sep="\t")
        else:
            pass


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


def gen_output(outf, group, exps, crapome):

    # protA protB format
    outname = os.path.join(outf, "adj_list_{}.txt".format(group))
    # outfile = gr_exps.multi_collapse()
    outfile = exps.to_adj_lst()
    # outfile.reset_index(inplace=True)
    outfile['confidence'] = outfile.apply(label_inte, axis=1)
    crap = io.read_crap(crapome)
    outfile['Frequency_crapome_ProtA'] = outfile['ProtA'].map(crap)
    outfile['Frequency_crapome_ProtB'] = outfile['ProtB'].map(crap)
    outfile.fillna(0, inplace=True)
    outfile.to_csv(os.path.join(outname), sep="\t", index=False)

    # prot centric output
    G = nx.from_pandas_edgelist(outfile, 'ProtA', 'ProtB')
    k = {}
    for node in G.nodes():
        tmp =  list(G.neighbors(node))
        k[node] = [", ".join(tmp), len(tmp)]
    reshaped = pd.DataFrame.from_dict(k, orient='index')
    reshaped.reset_index(inplace=True)
    reshaped.columns = ['Protein', 'interactors', '# interactors']
    reshaped = reshaped[reshaped['Protein'] != reshaped['interactors']]
    outname2 = os.path.join(outf, "prot_centr_{}.txt".format(group))
    reshaped.sort_values(by=['Protein'], inplace=True)
    reshaped.to_csv(os.path.join(outname2), sep="\t", index=False)


def estimate_background():
    """
    estimated background from crapome
    """
    pass


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
    exp_info = pd.read_csv(ids, sep="\t")
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    groups = dict(zip(exp_info['group'], exp_info['short_id']))
    allids = []
    gr_graphs = []
    for k, v in groups.items():
        gr_exps = NetworkCombiner()
        grids = []
        group_info = exp_info[exp_info['group']==k]
        for smpl in dir_:
            base = os.path.basename(os.path.normpath(smpl))
            fl = "./Input/{}.txt".format(base)
            if not group_info['Sample'].str.contains(fl).any():
                continue
            pred_out = os.path.join(smpl, "dnn.txt")
            grids.extend(list(pd.read_csv(fl, sep="\t")["GN"]))
            exp = TableConverter(table=pred_out, cond=pred_out)
            exp.fdr_control()
            exp.convert_to_network()
            exp.weight_adj_matrx(smpl, write=True)
            gr_exps.add_exp(exp)
        gr_exps.create_dfs()
        gr_exps.combine_graphs(grids)
        gr_exps.adj_matrix_multi()
        allprobs = gr_exps.multi_prob_out()
        allprobs.to_csv(os.path.join(outf, 'probtot_{}.txt'.format(v)), sep='\t')
        # group specific graph
        gr_graphs.append(gr_exps.get_gr_network())
        gen_output(outf, v, gr_exps, crapome)
        allids.extend(grids)

    G2 = gr_graphs.pop()
    mx = 0
    # here select max interaction score per ppi across baits
    for g in gr_graphs:
        for a,b, attrs in g.edges(data=True):
            if G2.has_edge(a,b):
                if G2[a][b]['weight'] < attrs['weight']:
                    G2[a][b]['weight'] = attrs['weight']
            else:
                G2.add_edge(a,b, weight=attrs['weight'])
    # filter weights
    G3 = nx.Graph()
    tokeep = [(a,b, attrs['weight']) for a, b, attrs in G2.edges(data=True) if attrs["weight"] >= 0.5]
    G3.add_weighted_edges_from(tokeep)
    todf = [[a,b,attr['weight']] for a,b,attr in G3.edges(data=True)]
    df = pd.DataFrame(todf, columns=['ProtA', 'ProtB', 'weight'])
    df.to_csv(os.path.join(tmp_, "comb_graph_adj.txt"),sep="\t")
    allids = sorted(list(set(G3.nodes)))
    alladj = nx.adjacency_matrix(G3, nodelist=allids, weight="weight").todense()
    np.savetxt(os.path.join(tmp_, "adj_mult.csv"), alladj, delimiter=",")
    with open(os.path.join(tmp_, "ids.pkl"), "wb") as f:
        pickle.dump(allids, f)
