import community
import networkx as nx
from karateclub.estimator import Estimator

class EgoNetSplitter(Estimator):
    r"""An implementation of `"Ego-Splitting" <https://www.eecs.yorku.ca/course_archive/2017-18/F/6412/reading/kdd17p145.pdf>`_
    from the KDD '17 paper "Ego-Splitting Framework: from Non-Overlapping to Overlapping Clusters". The tool first creates
    the ego-nets of nodes. A persona-graph is created which is clustered by the Louvain method. The resulting overlapping
    cluster memberships are stored as a dictionary.

    Args:
        resolution (float): Resolution parameter of Python Louvain. Default 1.0.
    """
    def __init__(self, resolution=1.0):
        self.resolution = resolution

    def _create_egonet(self, node):
        """
        Creating an ego net, extracting personas and partitioning it.

        Arg types:
            * **node** *(int)* - Node ID for ego-net (ego node).
        """
        ego_net_minus_ego = self.graph.subgraph(self.graph.neighbors(node))
        components = {i: n for i, n in enumerate(nx.connected_components(ego_net_minus_ego))}
        new_mapping = {}
        personalities = []
        for k, v in components.items():
            personalities.append(self.index)
            for other_node in v:
                new_mapping[other_node] = self.index
            self.index = self.index+1
        self.components[node] = new_mapping
        self.personalities[node] = personalities

    def _create_egonets(self):
        """
        Creating an ego-net for each node.
        """
        self.components = {}
        self.personalities = {}
        self.index = 0
        for node in self.graph.nodes():
            self._create_egonet(node)

    def _map_personalities(self):
        """
        Mapping the personas to new nodes.
        """
        self.personality_map = {p: n for n in self.graph.nodes() for p in self.personalities[n]}

    def _get_new_edge_ids(self, edge):
        """
        Getting the new edge identifiers.

        Arg types:
            * **edge** *(list of ints)* - Edge being mapped to the new identifiers.
        """
        return (self.components[edge[0]][edge[1]], self.components[edge[1]][edge[0]])

    def _create_persona_graph(self):
        """
        Create a persona graph using the ego-net components.
        """
        self.persona_graph_edges = [self._get_new_edge_ids(edge) for edge in self.graph.edges()]
        self.persona_graph = nx.from_edgelist(self.persona_graph_edges)

    def _create_partitions(self):
        """
        Creating a non-overlapping clustering of nodes in the persona graph.
        """
        self.partitions = community.best_partition(self.persona_graph, resolution=self.resolution)
        self.overlapping_partitions = {node: [] for node in self.graph.nodes()}
        for node, membership in self.partitions.items():
            self.overlapping_partitions[self.personality_map[node]].append(membership)

    def fit(self, graph):
        """
        Fitting an Ego-Splitter clustering model.

        Arg types:
            * **graph** *(NetworkX graph)* - The graph to be clustered.
        """
        self.graph = graph
        self._create_egonets()
        self._map_personalities()
        self._create_persona_graph()
        self._create_partitions()

    def get_memberships(self):
        r"""Getting the cluster membership of nodes.

        Return types:
            * **memberships** *(dictionary of lists)* - Cluster memberships.
        """
        return self.overlapping_partitions
