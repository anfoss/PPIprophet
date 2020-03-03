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
    '''
    Combine all replicates for a single condition into a network
    returns a network
    '''
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
        '''
        fill graphs with all proteins missing and weight 0 (i.e no connection)
        '''
        self.networks = [x.fill_graph(ids_all) for x in self.exps]
        return True

    def adj_matrix_multi(self):
        '''
        add sparse adj matrix to the adj_matrix container
        '''
        all_adj = []
        n = 0
        for G in self.networks:
            nd = list(map(str, G.nodes()))
            adj = nx.adjacency_matrix(
                                    G,
                                    nodelist=sorted(nd),
                                    weight='weight'
                                    )
            self.ids = sorted(nd)
            all_adj.append(adj.todense())
            n +=1
        # now multiply each element for the others
        self.adj_matrx = all_adj.pop()
        for m1 in all_adj:
            self.adj_matrx = np.multiply(self.adj_matrx, m1)
        return self.adj_matrx # / n

    def multi_collapse(self):
        self.combined = reduce(lambda x, y: pd.merge(x, y,
                                            on = ['ProtA', 'ProtB'],
                                            how='outer'),
                    self.dfs)
        self.combined.fillna(0, inplace=True)
        self.combined.set_index(['ProtA', 'ProtB'], inplace=True)
        self.combined['CombProb'] =  np.prod(self.combined.values, axis=1)
        return self.combined

    def get_ids(self):
        return self.ids

    def get_adj(self):
        return self.adj_matrx


class TableConverter(object):
    '''docstring for TableConverter'''
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
        # self.clean_name('ProtA')
        # self.clean_name('ProtB')
        for row in self.df.itertuples():
            self.G.add_edge(row[1], row[2], weight=row[3])
        return True

    def fill_graph(self, ids, w=10**-17):
        G2 = fully_connected(ids)
        [self.G.add_edge(*p, weight=w) for p in G2.edges() if not self.G.has_edge(p[0], p[1])]
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
                                        self.G,
                                        nodelist=sorted(map(str, self.G.nodes())),
                                        weight='weight'
                                        )
        self.adj = self.adj.todense()
        if write:
            nm = os.path.join(path, 'adj_matrix.txt')
            np.savetxt(nm, self.adj, delimiter="\t")
        return True

    def add_fdr(self):
        '''
        add fdr column to df. sorting is done due to different annotation
        in prob column (np vs pd)
        '''
        self.df = pd.merge(self.df, self.fdr[['prob', 'fdr']], left_on='Prob', right_on='prob', how='left')
        self.df.drop('prob', inplace=True, axis=1)
        self.df.fillna(0, inplace=True)
        return True

    def modify_df(self, fdr_thresh):
        '''
        substitute 0 to everything below fdr threshold
        '''
        self.df['Prob'].values[self.df['Prob'] <= fdr_thresh] = 0

    def get_adj_matrx(self):
        return self.adj

    def get_df(self):
        return self.df

    def calc_fdr(self, path, bait='UXT', target_fdr=0.2):
        '''
        get shell level of interaction of bait and then calc local fdr
        for every pred level
        '''
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
        # neg = neg[neg['Prob']>=0.5]
        pos, neg = pos['Prob'].values, neg['Prob'].values
        scores =  np.unique(np.concatenate((pos, neg)))
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
        # plot_fdr(pos, neg, fdr, path)
        fdr_df = pd.DataFrame(fdr, columns=['target', 'decoy', 'fdr', 'prob'])
        fdr_df.to_csv(os.path.join(path, 'fdr.txt'), index=False, sep="\t")
        fdr_thresh = 0.5
        try:
            # either first value smaller than target
            fdr_thresh = fdr_df[fdr_df['fdr'] <= target_fdr]['prob'].values[0]
        except IndexError as e:
            # or first value bigger than target
            fdr_thresh = fdr_df[fdr_df['fdr'] >=target_fdr]['prob'].values[-1]
        print('Estimated prob threshold for {} is {}'.format(path, fdr_thresh))
        self.fdr = fdr_df
        self.modify_df(fdr_thresh)
        self.add_fdr()


def plot_fdr(target_dist, decoy_dist, fdr, path):
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot(311)
    binNum = 100.0
    dist = np.unique(np.concatenate((target_dist, decoy_dist)))
    binwidth = (max(dist) - min(dist)) / binNum

    plt.hist(target_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='r', edgecolor='r', alpha=0.3, label='Target')
    plt.hist(decoy_dist, bins=np.arange(min(dist), max(dist) + binwidth, binwidth),
             color='b', edgecolor='b', alpha=0.3, label='Decoy')
    plt.ylabel('Frequency')
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(fdr['prob'], fdr['target'], 'r', label='P[Target>x]')
    plt.plot(fdr['prob'], fdr['decoy'], 'b--', label='P[Decoy>x]')
    plt.plot(fdr['prob'], fdr['fdr'], 'k', label='FDR')
    plt.xlabel('DNN probability')
    plt.legend()
    box = ax2.get_position()
    ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    ax3 = plt.subplot(313)
    plt.plot(fdr['decoy'], fdr['target'], 'k')
    plt.ylabel('True positive rate')
    plt.xlabel('False positive rate')
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    plt.tight_layout()
    outfile = os.path.join(path, 'fdr.pdf')
    plt.savefig(outfile, dpi=800, bbox_inches='tight')
    return True


def fully_connected(l, w=10**-17):
    G = nx.Graph()
    [G.add_edge(u,q, weight=w) for u,q in itertools.combinations(l,2)]
    return G


def runner(tmp_, ids, outf):
    '''
    read folder tmp in directory.
    then loop for each file and create a combined file which contains all files
    creates in the tmp directory
    '''
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
        pred_out = os.path.join(smpl, 'dnn.txt')
        raw_matrix = os.path.join(smpl, 'transf_matrix.txt')
        allids.extend(list(pd.read_csv(raw_matrix, sep="\t")['ID']))
        exp = TableConverter(
            name=exp_info[base],
            table=pred_out,
            cond=pred_out
        )
        exp.convert_to_network()
        exp.weight_adj_matrx(smpl, write=True)
        allexps.add_exp(exp)
    allexps.create_dfs()
    allexps.combine_graphs(allids)
    m_adj = allexps.adj_matrix_multi()
    ids = allexps.get_ids()
    with open(os.path.join(tmp_, 'ids.pkl'), 'wb') as f:
        pickle.dump(ids, f)

    # protA protB format
    outname = os.path.join(outf, 'adj_list.txt')
    outfile = allexps.multi_collapse()
    outfile.to_csv(os.path.join(outname), sep="\t", index=True)

    # adj matrix
    np.savetxt(os.path.join(tmp_, 'adj_mult.csv'), m_adj, delimiter=',')
    # m_adj = pd.DataFrame(m_adj, index=ids)
    # m_adj.columns = ids
    # m_adj.to_csv(os.path.join(outf, 'adj_matrix_combined.txt'), sep="\t")
