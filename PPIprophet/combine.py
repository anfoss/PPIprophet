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
    NetworkCombiner Class

    This class is designed to combine multiple replicate networks for a single condition into a single network. 
    It provides methods for adding experiments, creating dataframes, combining graphs, calculating probabilities, 
    and generating adjacency matrices and frequency matrices.

    Attributes:
        exps (list): A list of experiment objects to be combined.
        adj_matrx (numpy.ndarray): The combined adjacency matrix.
        networks (list): A list of individual network graphs.
        dfs (list): A list of dataframes corresponding to the experiments.
        ids (list): A sorted list of node IDs.
        combined (pandas.DataFrame): A combined dataframe of probabilities.
        combine (str): The method used for combining networks. Default is "prob".
        comb_net (networkx.Graph): The combined network graph.
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
        self.combine = "prob"
        self.comb_net = None

    def add_exp(self, exp):
        self.exps.append(exp)

    def create_dfs(self):
        [self.dfs.append(x.get_df()) for x in self.exps]

    def combine_graphs(self, ids_all):
        """
        Combines graphs by filling them with all proteins that are missing 
        and assigning a weight of 0 (indicating no connection).

        Args:
            ids_all (list): A list of all protein identifiers to ensure 
                            all graphs are filled with the same set of proteins.

        Returns:
            bool: Always returns True to indicate the operation was completed.
        fill graphs with all proteins missing and weight 0 (i.e no connection)
        """
        self.networks = [x.fill_graph(ids_all) for x in self.exps]
        return True

    def multi_prob_out(self):
        """
        Combines edge probabilities from multiple network graphs into a single DataFrame.

        This method processes a list of network graphs, extracts their edge lists with 
        associated weights, and combines them into a unified DataFrame. The resulting 
        DataFrame contains all edges from the input graphs, with probabilities merged 
        across the graphs.

        Returns:
            pandas.DataFrame: A DataFrame with columns ["source", "target", "weight1", "weight2", ...],
            where "source" and "target" represent the nodes of an edge, and "weightX" represents 
            the edge weight (probability) from the X-th graph. Rows with all NaN weights are dropped.
        """
        allprobs = []
        for i, G in enumerate(self.networks):
            df = nx.to_pandas_edgelist(G)
            df2 = df.copy()
            df2.columns = ["target", "source", "weight"]
            df2 = df2[["source", "target", "weight"]]
            df = pd.concat([df, df2])
            df = df.groupby(["source", "target"], as_index=False).sum()
            df.rename(columns={"weight": f"weight_{i}"}, inplace=True)
            allprobs.append(df)

        tots = reduce(lambda l, r: pd.merge(l, r, on=["source", "target"], how="outer"), allprobs)
        tots = tots[tots['source']!= tots['target']]
        tots = tots.set_index(["source", "target"])
        tots = tots.replace(10**-17, np.nan)
        tots.dropna(how="all", inplace=True)
        return tots


    def calc_freq(self, all_adj):
        """
        Calculate the frequency matrix from a list of adjacency matrices.
        This method processes a list of adjacency matrices, binarizes them based on a threshold 
        (values >= 0.5 are set to 1, others to 0), and computes the frequency of connections 
        across all matrices.
        Args:
            all_adj (list of numpy.ndarray): A list of adjacency matrices where each matrix 
                                             is a 2D numpy array.
        Returns:
            numpy.ndarray: A 2D numpy array representing the frequency matrix, where each 
                           element is the mean value of the corresponding elements across 
                           the binarized adjacency matrices.
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
        return np.exp(
            (w_freq * np.log(freq) + w_prob * np.log(prob)) / (w_freq + w_prob)
        )

    def adj_matrix_multi(self):
        """
        Combines adjacency matrices from multiple networks based on the specified combination method.
        This method processes a list of networks, extracts their adjacency matrices, and combines them
        using one of the following methods:
        - "prob": Takes the element-wise maximum of all adjacency matrices.
        - "comb": Multiplies the probabilities element-wise across all adjacency matrices.
        - "mean": Computes the element-wise mean of all adjacency matrices.
        The resulting combined adjacency matrix is then converted into a networkx graph.
        Returns:
            numpy.ndarray: The combined adjacency matrix.
        Attributes:
            self.ids (list): A sorted list of node identifiers, ensuring consistent node order across networks.
            self.networks (list): A list of networkx graphs to be combined.
            self.combine (str): The method used to combine adjacency matrices ("prob", "comb", or "mean").
            self.adj_matrx (numpy.ndarray): The resulting combined adjacency matrix.
            self.comb_net (networkx.Graph): The resulting combined networkx graph.
        """        
        # all_adj = []
        # # enforce same order
        # self.ids = sorted(list(map(str, self.networks[0].nodes())))
        # for G in self.networks:
        #     adj = nx.adjacency_matrix(G, nodelist=self.ids, weight="weight")
        #     all_adj.append(adj.todense())
        # if self.combine == "prob":
        #     # adj matrix with max prob (no filter)
        #     self.adj_matrx = np.maximum.reduce(all_adj)
        # elif self.combine == "comb":
        #     # now multiply probabilities
        #     self.adj_matrx = all_adj.pop()
        #     for m1 in all_adj:
        #         self.adj_matrx = np.multiply(self.adj_matrx, m1)
        #         # self.adj_matrx = np.add(self.adj_matrx, m1)
        # elif self.combine == "mean":
        #     self.adj_matrx = all_adj.pop()
        #     for m1 in all_adj:
        #         self.adj_matrx = np.add(self.adj_matrx, m1)
        #     self.adj_matrx = self.adj_matrx / (len(all_adj) + 1)
        # # calculate frequency
        # # freq = self.calc_freq(all_adj)
        # # self.adj_matrx = self.desi_f(freq, self.adj_matrx)
        # # convert to network and return
        # # nodes are alphabetically ordered
        # self.comb_net = nx.from_numpy_array(self.adj_matrx)
        # self.comb_net = nx.relabel_nodes(
        #     self.comb_net, dict(zip(self.comb_net.nodes, self.ids), copy=False)
        # )
        # return self.adj_matrx
        all_ids = sorted(set().union(*[map(str, G.nodes()) for G in self.networks]))
        self.ids = all_ids  # Store for later use
        
        # Build aligned adjacency matrices (fill missing nodes with 0s)
        all_adj = []
        for G in self.networks:
            adj = nx.to_numpy_array(G, nodelist=self.ids, weight="weight", nonedge=0.0)
            all_adj.append(adj)

        all_adj = np.stack(all_adj)  # shape: (n_networks, n_nodes, n_nodes)

        if self.combine == "prob":
            self.adj_matrx = np.max(all_adj, axis=0)
        elif self.combine == "comb":
            self.adj_matrx = np.prod(all_adj, axis=0)
        elif self.combine == "mean":
            self.adj_matrx = np.mean(all_adj, axis=0)
        else:
            raise ValueError(f"Unknown combine method: {self.combine}")

        self.comb_net = nx.from_numpy_array(self.adj_matrx)
        self.comb_net = nx.relabel_nodes(
            self.comb_net, dict(zip(self.comb_net.nodes, self.ids)), copy=False
        )
        return self.adj_matrx


    def to_adj_lst(self):
        """
        converts adjacency matrix to adj list
        """
        df = nx.to_pandas_edgelist(self.comb_net)
        df.columns = ["ProtA", "ProtB", "CombProb"]
        df = df[df["ProtA"] != df["ProtB"]]
        df = df[df["CombProb"] >= 0.5]
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
        G = nx.from_numpy_array(self.adj_matrx)
        G = nx.relabel_nodes(G, dict(zip(G.nodes, self.ids)))
        return G


class TableConverter(object):
    """
    TableConverter is a utility class for converting protein-protein interaction tables into network representations,
    applying FDR control, and generating adjacency matrices for downstream analysis.

    Attributes:
        df (pd.DataFrame): Input dataframe containing protein interaction data.
        cond (str): Experimental condition o.
        G (networkx.Graph): Graph representation of the protein interactions.
        adj (np.ndarray): Weighted adjacency matrix of the graph.
        cutoff_fdr (float): False discovery rate cutoff for filtering interactions.

    Methods:
        __init__(df, cond, fdr):
            Initializes the TableConverter with a dataframe, condition, and FDR cutoff.

        clean_name(col):
            Cleans protein names in the specified column by removing suffixes after underscores.

        convert_to_network():
            Converts the dataframe into a networkx Graph, adding edges with weights.

        fill_graph(ids, w=10**-17):
            Ensures the graph is fully connected by adding missing edges with a small weight.

        weight_adj_matrx(path, write=True):
            Generates and optionally writes the weighted adjacency matrix of the graph to a file.

        fdr_control(plot=True):
            Applies FDR control to filter interactions based on probability and writes error statistics.

        get_adj_matrx():
            Returns the weighted adjacency matrix.

        get_df():
            Returns the filtered dataframe.
    """

    def __init__(self, path, df, cond, fdr, cutoff_fdr=0.75):
        super(TableConverter, self).__init__()
        self.table = path
        self.df = df        
        self.cond = cond
        self.G = nx.Graph()
        self.adj = None
        # used 0.75 in paper
        self.fdr = float(fdr)
        self.cutoff_fdr = 0.75

    def clean_name(self, col):
        self.df[col] = self.df[col].str.split("_").str[0]

    def convert_to_network(self):
        # self.clean_name('ProtA')
        # self.clean_name('ProtB')
        for row in self.df.itertuples():
            self.G.add_edge(row[1], row[2], weight=row[3])
        return True

    def fill_graph(self, ids, w=10**-17):
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

    def fdr_control(self, plot=True):
        self.df = self.df[self.df["Prob"] >= 0.5]
        if self.fdr < 1:
            decoy = self.df[self.df["isdecoy"] == "DECOY"]["Prob"].values
            target = self.df[self.df["isdecoy"] == "TARGET"]["Prob"].values
            error_stat = qvalue.error_statistics(target, decoy)
            i0 = (error_stat.qvalue - self.fdr).abs().idxmin()
            self.cutoff_fdr = error_stat.iloc[i0]["cutoff"]
            print("Probability cutoff for reaching {} FDR in {} is {}".format(self.fdr, self.table, self.cutoff_fdr))
            # self.df = self.df[self.df['Prob'] >= self.cutoff_fdr]
            # 0s the probability below fdr thresholds
            self.df[self.df["Prob"] <= self.cutoff_fdr] = 10**-17
            self.df = self.df[self.df["isdecoy"] == "TARGET"]
            pth = os.path.dirname(os.path.abspath(self.table))
            error_stat.to_csv("{}/error_metrics.txt".format(pth), sep="\t")
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
    if subs["CombProb"] > 0.9:
        return "High confidence"
    elif subs["CombProb"] > 0.75:
        return "Medium confidence"
    elif subs["CombProb"] >= 0.5:
        return "Low confidence"
    else:
        return "No interaction"


def gen_output(outf, group, exps):

    # protA protB format
    outname = os.path.join(outf, "adj_list_{}.txt".format(group))
    # outfile = gr_exps.multi_collapse()
    outfile = exps.to_adj_lst()
    # outfile.reset_index(inplace=True)
    outfile["confidence"] = outfile.apply(label_inte, axis=1)
    outfile.fillna(0, inplace=True)
    outfile.to_csv(os.path.join(outname), sep="\t", index=False)

    # prot centric output
    G = nx.from_pandas_edgelist(outfile, "ProtA", "ProtB")
    k = {}
    for node in G.nodes():
        tmp = list(G.neighbors(node))
        k[node] = [", ".join(tmp), len(tmp)]
    reshaped = pd.DataFrame.from_dict(k, orient="index")
    reshaped.reset_index(inplace=True)
    reshaped.columns = ["Protein", "interactors", "# interactors"]
    reshaped = reshaped[reshaped["Protein"] != reshaped["interactors"]]

    outname2 = os.path.join(outf, "prot_centr_{}.txt".format(group))
    reshaped.sort_values(by=["Protein"], inplace=True)
    reshaped.to_csv(os.path.join(outname2), sep="\t", index=False)


def estimate_background():
    """
    estimated background from crapome
    """
    pass


def runner(tmp_, ids, outf, fdr):
    """
    Processes experimental data, combines graphs, and generates output files.
    Args:
        tmp_ (str): Path to the temporary directory containing intermediate files.
        ids (str): Path to a tab-separated file containing experiment information, 
                    including group and sample details.
        outf (str): Path to the output directory where results will be saved.
    Workflow:
        1. Reads experiment information from the `ids` file.
        2. Groups experiments by the 'group' column in the `ids` file.
        3. For each group:
            - Reads and processes sample files.
            - Converts data into network representations.
            - Combines graphs and computes adjacency matrices.
            - Filters edges based on weight thresholds.
            - Generates group-specific output files, including:
                - Probability files.
                - Combined graph files.
                - Adjacency matrices.
                - Node ID pickle files.
    Output:
        - Group-specific probability files: "AllProbabilityRepl_<group_name>.txt"
        - Combined graph files: "CombinedGraph_AllSamples_<group_name>.txt"
        - Adjacency matrices: "adj_mult_<group_name>.csv"
        - Node ID pickle files: "ids_<group_name>.pkl"
    Notes:
        - The function assumes specific file structures and naming conventions for input files.
        - Filters edges with weights below 0.85 when creating the final graph.
        - Uses NetworkX for graph operations and pandas for data manipulation.
    Raises:
        - FileNotFoundError: If input files or directories are missing.
        - ValueError: If data formatting issues are encountered.
    """

    if not os.path.isdir(outf):
        os.makedirs(outf)
    dir_ = []
    dir_ = [x[0] for x in os.walk(tmp_) if x[0] is not tmp_]
    exp_info = pd.read_csv(ids, sep="\t")
    strip = lambda x: os.path.splitext(os.path.basename(x))[0]
    groups = dict(zip(exp_info["group"], exp_info["short_id"]))
    allids = []
    gr_graphs = []
    for cnd, grp_info in exp_info.groupby('cond'):
        allids = []
        gr_graphs = []
        gr_exps = NetworkCombiner()
        grids = []
        for flname in grp_info['Sample']:
            # this formats to keep only /tmp/xx -> xx which is the 
            base = os.path.basename(os.path.normpath(flname))
            tmp_filepath = os.path.join(tmp_, base.replace('.txt', ''))
            pred_out = os.path.join(tmp_filepath, "dnn.txt")
            tmp = list(pd.read_csv(flname, sep='\t')["GN"])
            grids.extend(tmp)
            exp = TableConverter(path=pred_out, df=pd.read_csv(pred_out, sep='\t'), cond=cnd, fdr=fdr)
            exp.fdr_control(plot=True)
            exp.convert_to_network()
            exp.weight_adj_matrx(tmp_filepath, write=True)
            gr_exps.add_exp(exp)
        
        gr_exps.create_dfs()
        gr_exps.combine_graphs(grids)
        gr_exps.adj_matrix_multi()
        allprobs = gr_exps.multi_prob_out()
        grp_name = grp_info['short_id'].values[0]
        print(allprobs)
        # allprobs = allprobs[~allprobs["weight"].isna()]
        allprobs.to_csv(os.path.join(outf, "probtot_{}.txt".format(grp_name)), sep="\t")
        # group specific graph
        gr_graphs.append(gr_exps.get_gr_network())
        gen_output(outf, cnd, gr_exps)
        allids.extend(grids)
        
    G2 = gr_graphs.pop()
    # here select max interaction score per ppi across baits
    for g in gr_graphs:
        for a, b, attrs in g.edges(data=True):
            if G2.has_edge(a, b):
                if G2[a][b]["weight"] < attrs["weight"]:
                    G2[a][b]["weight"] = attrs["weight"]
            else:
                G2.add_edge(a, b, weight=attrs["weight"])
    # filter weights
    G3 = nx.Graph()
    tokeep = [
        (a, b, attrs["weight"])
        for a, b, attrs in G2.edges(data=True)
        if attrs["weight"] >= 0.5
    ]
    G3.add_weighted_edges_from(tokeep)
    todf = [[a, b, attr["weight"]] for a, b, attr in G3.edges(data=True)]
    df = pd.DataFrame(todf, columns=["ProtA", "ProtB", "weight"])
    df.to_csv(os.path.join(tmp_, "comb_graph_adj.txt"), sep="\t")
    allids = sorted(list(set(G3.nodes)))
    alladj = nx.adjacency_matrix(G3, nodelist=allids, weight="weight").todense()
    np.savetxt(os.path.join(tmp_, "adj_mult.csv"), alladj, delimiter=",")
    with open(os.path.join(tmp_, "ids.pkl"), "wb") as f:
        pickle.dump(allids, f)
