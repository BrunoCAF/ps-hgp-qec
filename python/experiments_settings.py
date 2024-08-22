import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy.special import comb

import networkx as nx
import networkx.algorithms.bipartite as bpt

path_to_initial_codes = '../initial_codes/'
codes = ['[625,25]', '[1225,65]', '[1600,64]', '[2025,81]']
textfiles = [f"HGP_(3,4)_{code}.txt" for code in codes]

state_space_params = [(15, 20, 60), 
                      (21, 28, 84), 
                      (24, 32, 96), 
                      (27, 36, 108)]

MC_budget = int(1e4)
noise_levels = [9/32, 8/32, 9/32, 12/32]
# times: 15, 40, 80, 200

def load_tanner_graph(filename: str) -> nx.MultiGraph:
    m, n = np.loadtxt(filename, max_rows=1, dtype=int)
    indices, indptr = np.array([], dtype=int), [0]
    for r in range(m):
        r_ind = np.loadtxt(filename, skiprows=r+1, max_rows=1, dtype=int)
        indices = np.concatenate([indices, np.sort(r_ind)])
        indptr.append(len(r_ind))
    
    H = sp.csr_array((m, n), dtype=int)
    H.data = np.ones_like(indices, dtype=int)
    H.indices = indices
    H.indptr = np.cumsum(indptr)

    return bpt.from_biadjacency_matrix(H, create_using=nx.MultiGraph)


def parse_edgelist(state: nx.MultiGraph) -> np.ndarray:
    return np.array(sorted(state.edges(data=False)), dtype=np.uint8).flatten() # shape: (2*E,)


def from_edgelist(edgelist: np.ndarray) -> nx.MultiGraph:
    diam = lambda arr: np.max(arr) - np.min(arr) + 1
    m, n = np.apply_along_axis(diam, 0, edgelist.reshape(-1,2))

    B = nx.MultiGraph()
    B.add_nodes_from(np.arange(m), bipartite=0)
    B.add_nodes_from(np.arange(m, m+n-1), bipartite=1)
    B.add_edges_from([tuple(r) for r in edgelist.reshape(-1, 2)])

    return B


def generate_neighbor(theta: nx.MultiGraph) -> nx.MultiGraph:
    # Copy state
    neighbor = nx.MultiGraph(theta)
    
    # get (multi)edge number from state theta
    E = neighbor.number_of_edges()

    # compute action space size
    A = comb(E, 2, exact=True)
    
    # sample action
    a = npr.choice(A)
    
    # convert to edge indices
    i = np.floor(((2*E - 1) - np.sqrt((2*E-1)**2 - 8*a))//2).astype(int)
    j = (a - E*i + ((i+2)*(i+1))//2)
    
    # apply cross-wiring 
    edge_list = sorted(neighbor.edges(data=False))
    e1, e2 = edge_list[i], edge_list[j]
    (c1, n1), (c2, n2) = e1, e2
    f1, f2 = (c1, n2), (c2, n1)
    neighbor.remove_edges_from([e1, e2])
    neighbor.add_edges_from([f1, f2])
    
    return neighbor
