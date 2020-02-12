import numpy as np
import networkx as nx
from copy import deepcopy
from itertools import product
from scipy import sparse

def specialize(A, base):
    """
    Specialize network around a base set according to the specialization model
    described in:
        Spectral and Dynamic Consequences of Network Specialization
        https://arxiv.org/abs/1908.04435

    Parameters
    ----------
    A: (Adjacency matrix) May be an  scipy.sparse matrix or ndarray
    base: List of nodes

    """

    base = list(base)
    lg = LightGraph(A)
    spec, origin = specialize_light_graph(lg,base)

    if sparse.issparse(A):
        S = spec.sparse_adj()
        S.astype(A.dtype)
        return S, origin
    else:
        return spec.adj(), origin

class LightGraph(object):
    """
    A lightweight graph class for the specialization algorithm.

    Attributes:
    -----------
    self.edges (dict): Associates edges with their respective weights.
                       Note that in LightGraph (0,1) represents an edge
                       from node 1 to node 0, unlike networkx graphs.

                       (e.g.) self.edges = {(0,1):.5, (2,0):.1} would
                       represent an edge weight of .5 on the edge from
                       node 1 to node 0 and an edge weight of .1 on the
                       edge from node 0 to node 2.

    self.n (int):      Number of nodes

    self.orign (list): The value self.orign[i] is the origin of node i.
                       Unless the graph was specialized, self.origin[i] = i.
    """
    #---------------------------------
    # Initialization functions
    #---------------------------------

    def __init__(self, *args):
        if len(args) == 2:
            self.edges = args[0]
            self.n     = args[1]
            self.origin = list(range(self.n))
        elif len(args) == 1:
            args = args[0]
            if sparse.issparse(args):
                A = args.todok()
                self._init_from_dok(A)
            if type(args) == np.ndarray:
                self._init_from_adj(args)
            if type(args) == nx.DiGraph:
                self._init_from_DiGraph(args)

    def __repr__(self):
        return "LightGraph(%s nodes, %s edges)" % (self.n, len(self.edges))

    def _init_from_adj(self, A):
        y,x        = np.where(A != 0)
        self.edges = {e:A[e] for e in list(zip(y,x))}
        self.n     = A.shape[0]
        self.origin = list(range(self.n))


    def _init_from_dok(self, A):
        self.edges = {**A}
        self.n = A.shape[0]
        self.origin = list(range(self.n))

    def _init_from_DiGraph(self, G):
        self.edges = {(e[1],e[0]): G[e[0]][e[1]].get("weight", 1.0) for e in G.edges}
        self.n     = len(G.nodes)
        self.origin = list(range(self.n))

    #---------------------------------
    # Send LightGraph to new format
    #---------------------------------

    def digraph(self):
        G = nx.DiGraph()
        G.add_nodes_from(range(self.n))
        G.add_weighted_edges_from([(e[1],e[0],w) for e,w in self.edges.items()])
        return G

    def adj(self):
        A = np.zeros((self.n, self.n))
        for e,w in self.edges.items():
            A[e] = w
        return A

    def sparse_adj(self):
        A = sparse.dok_matrix((self.n,self.n))
        A._update(self.edges)
        return A

    #---------------------------------
    # Function used for specialization
    #---------------------------------

    def _add_scc(self, compnt, edges, weight_dict, e, e_weight ):
        """ Adds a strongly connected component and one edge
            from the rest of the graph to the scc
        """
        # New node labels and origins
        labels = dict()
        for i,c in enumerate(compnt):
            labels[c] = i+self.n
            self.origin.append(c)

        # Add edges in scc
        for i,j in edges:
            self.edges[(labels[i],labels[j])] = weight_dict[(i,j)]
        # Add edge from the graph to scc
        self.edges[(labels[e[0]],e[1])] = e_weight

        self.n += len(compnt)
        return labels

# end class

def specialize_light_graph(lg,base):

    spec_set = set(range(lg.n)) - set(base) # Find the nodes to be specialized
    edges = list(lg.edges.keys())

    # Begin specialized graph
    new_lg = LightGraph(dict(),len(base))
    base_labels = dict()
    for i,b in enumerate(base):
        base_labels[b] = i
        new_lg.origin[i] = b


    # Sort Edges
    edge_bins = {"s2s":[], "b2s":[], "s2b":[]}
    for i,e in enumerate(edges):
        n,m = e
        if n in spec_set:
            if m in spec_set:
                edge_bins["s2s"].append(i)
            else:
                edge_bins["b2s"].append(i)
        else:
            if m in spec_set:
                edge_bins["s2b"].append(i)
            else:
                new_lg.edges[(base_labels[n],base_labels[m])] = lg.edges[e]


    # Find scc in the specialization set
    G_spec = nx.DiGraph()
    G_spec.add_nodes_from(spec_set)
    G_spec.add_edges_from([edges[i][::-1] for i in edge_bins["s2s"]])
    compnts = [list(c) for c in nx.strongly_connected_components(G_spec)]
    # Make a mapping that sends nodes to their component
    comp_labels = dict()
    for i,c in enumerate(compnts):
        for n in c:
            comp_labels[n]=i

    # Find all transition edges between components
    trans_edges = dict()
    comp_edges = [[] for c in compnts]

    for i,j in [edges[k] for k in edge_bins["s2s"]]:
        n,m = comp_labels[i],comp_labels[j]
        if m != n:
            # Check if the nodes are already in the dict
            if not(n in trans_edges):
                trans_edges[n] = dict()
            if not (m in trans_edges[n]):
                trans_edges[n][m] = []
            # Record an edge between the components
            trans_edges[n][m].append((i,j))
        else:
            # Store intercomponent edges
            comp_edges[comp_labels[i]].append((i,j))

    # Compress the specialization set onto the scc
    G_compr = nx.DiGraph()
    G_compr.add_nodes_from(range(len(compnts)))
    for i in trans_edges.keys():
        for j in trans_edges[i].keys():
            G_compr.add_edge(j,i)

    # Find all branches through the compressed graph:

    # Find the node heirarchy (because it is acyclic)
    topo_sort = nx.topological_sort(G_compr)
    all_branches = []
    npaths = -1
    # Make structures for storing where branches begin and end
    begins_at = {i:set() for i in range(len(compnts))}
    ends_at = {i:set() for i in range(len(compnts))}

    # Build the branches based on already found branches
    for n in reversed(list(topo_sort)):
        npaths += 1
        all_branches.append([n])
        begins_at[n].add(npaths)
        ends_at[n].add(npaths)

        for j in G_compr.neighbors(n):
            for k in begins_at[j]:
                npaths += 1
                path = all_branches[k]
                all_branches.append([n] + path)
                begins_at[n].add(npaths)
                ends_at[path[-1]].add(npaths)

    for i in edge_bins["b2s"]:
        for j in edge_bins["s2b"]:

            # Get and relabel ingoing edge and find the component it points to
            n,m = edges[i]
            e_b2s = (n, base_labels[m])
            beg_c = comp_labels[n]

            # Find which component will have to be the last in the branch
            n,m = edges[j]
            end_c = comp_labels[m]

            for idx in begins_at[beg_c].intersection(ends_at[end_c]):
                br = all_branches[idx]
                # Find every combination of edges between the components. There is some inefficiency here.
                edges_btw = [trans_edges[br[i+1]][br[i]] for i in range(len(br)-1)]
                edge_combs = [list(ec) for ec in product(*edges_btw)]

                labels = dict()
                for ec in edge_combs:
                    for k,c in enumerate(br):
                        if k == 0:
                            labels = new_lg._add_scc(compnts[c], comp_edges[c], lg.edges, e_b2s, lg.edges[edges[i]])
                        else:
                            n,m = ec[k-1]
                            e = (n, labels[m])
                            labels = new_lg._add_scc(compnts[c], comp_edges[c], lg.edges, e, lg.edges[ec[k-1]])
                    n,m = edges[j]
                    e_s2b = (base_labels[n],labels[m])
                    new_lg.edges[e_s2b] = lg.edges[edges[j]]

    return new_lg, new_lg.origin
