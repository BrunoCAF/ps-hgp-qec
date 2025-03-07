import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy.special import comb

import networkx as nx
import networkx.algorithms.bipartite as bpt
import pym4ri

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
p_vals = np.linspace(0.1, 0.5, 15)

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
    B.add_nodes_from(np.arange(m, m+n), bipartite=1)
    B.add_edges_from([tuple(r) for r in edgelist.reshape(-1, 2)])

    return B



def generate_neighbor_highlight(theta: nx.MultiGraph) -> tuple[nx.MultiGraph, tuple]:
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
    
    return neighbor, [e1, e2], [f1, f2]


def generate_neighbor(theta: nx.MultiGraph) -> nx.MultiGraph:
    neighbor, *_ = generate_neighbor_highlight(theta)
    return neighbor

def break_SS(theta: nx.MultiGraph) -> nx.MultiGraph:
    m = len(checks := [v for v, b in theta.nodes(data='bipartite') if b == 0])
    n = len(bits := [v for v, b in theta.nodes(data='bipartite') if b == 1])
    
    _, __, SSlist = smallest_SS_weight(bpt.biadjacency_matrix(theta, row_order=sorted(checks), column_order=sorted(bits)))
    ss = npr.choice(SSlist)

    # Build the bipartite graph
    G = nx.MultiGraph(theta)
    # Extract the erasure subgraph
    Gss = G.edge_subgraph([(u, v, w) for u, v, w in G.edges if (u, v) in G.edges(nbunch=m+np.nonzero(i2set(ss, n))[0])])
    
    # Select the check with smallest degree in the subgraph
    frozen_checks = [n for n, b in Gss.nodes(data='bipartite') if b == 0]
    fchk = min(frozen_checks, key=lambda x: Gss.degree(x))
    
    # Extract the non-erasure subgraph, without the selected check
    Gssbar = G.edge_subgraph([(u, v, w) for u, v, w in G.edges if (u, v) in G.edges(nbunch=m+np.nonzero(1-i2set(ss, n))[0])])
    Gssbar = Gssbar.subgraph([n for n in Gssbar.nodes if n != fchk])

    # Pick random edges to braid
    e1 = list(Gssbar.edges)[npr.choice(Gssbar.number_of_edges())]
    e2 = [(u, v, w) for u, v, w in Gss.edges if (u, v) in Gss.edges(fchk)][npr.choice(Gss.degree(fchk))]

    # Cross edges
    (u1, v1, _), (u2, v2, _) = e1, e2
    c1, n1 = min(u1, v1), max(u1, v1)
    c2, n2 = min(u2, v2), max(u2, v2)
    f1, f2 = (c1, n2), (c2, n1)
    
    # Remove selected edges and add the crossed edges
    G.remove_edges_from([e1, e2])
    G.add_edges_from([f1, f2])

    # Convert back to pcm
    Hprime = bpt.biadjacency_matrix(G, row_order=np.arange(m), column_order=np.arange(m, m+n))
    return Hprime, [e1, e2], [f1, f2]

import numba

@numba.jit(nopython=True)
def gosper_next(c):
    a = c & -c
    b = c + a
    return (((c ^ b) >> 2) // a) | b

@numba.jit(nopython=True)
def _smallest_SS_weight(H: np.ndarray) -> tuple[int, int, list]:
    _, n = H.shape
    found = False
    min_w, num_ss = 0, 0
    min_SS = []

    for weight in range(2, n):
        c = (1 << weight) - 1  # Smallest subset of size 'weight'
        while c < (1 << n):
            # Convert 'c' to a binary representation as a NumPy array
            candidate_ss = np.array([(c >> i) & 1 for i in range(n)], dtype=np.float32)

            # Check if it is a stopping set
            if np.count_nonzero((H @ candidate_ss) == 1) == 0:
                found = True
                min_w = weight
                num_ss += 1
                min_SS.append(c)

            c = gosper_next(c)  # Get next subset using Gosper's hack

        if found:
            return min_w, num_ss, min_SS

def smallest_SS_weight(H: sp.csr_array) -> tuple[int, int, list]:
    return _smallest_SS_weight((H.astype(np.uint8).todense()&1).astype(np.float32))

@numba.jit(nopython=True)
def _code_distance(H: np.ndarray) -> int:
    _, n = H.shape

    for weight in range(1, n):
        c = (1 << weight) - 1  # Smallest subset of size 'weight'
        while c < (1 << n):
            # Convert 'c' to a binary representation as a NumPy array
            candidate_cw = np.array([(c >> i) & 1 for i in range(n)], dtype=np.float32)

            # Check if it is a codeword
            if not np.any((H @ candidate_cw) % 2):
                return weight

            c = gosper_next(c)  # Get next subset using Gosper's hack


def code_distance(H: sp.csr_array) -> int:
    d = _code_distance((H.astype(np.uint8).todense()&1).astype(np.float32))
    return np.inf if d is None else d

def i2set(i: int, width: int) -> np.ndarray:
    return np.array([(i >> k) & 1 for k in range(width)], dtype=np.int8)

def from_parity_check_matrix(H: sp.csr_array) -> nx.MultiGraph:
    return bpt.from_biadjacency_matrix(H, create_using=nx.MultiGraph)

def code_dimension(H: sp.csr_array) -> int:
    _, n = H.shape
    return n - pym4ri.rank(H.astype(bool).todense())