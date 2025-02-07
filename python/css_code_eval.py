import numpy as np
import numpy.random as npr
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp

import pym4ri as m4ri

from tqdm import tqdm

# TODO: implement some strategy for importance sampling

def CSS_HGP_code_from_state(state: nx.MultiGraph) -> tuple[sp.sparray, int, int]:
    # Convert multigraph state to simple graph
    G = nx.Graph(state)
    
    # Extract biadjacency matrix from the Tanner graph of the classical code
    c = [n for n, b in G.nodes(data='bipartite') if b == 0]
    v = [n for n, b in G.nodes(data='bipartite') if b == 1]
    H = bpt.biadjacency_matrix(G, row_order=sorted(c), column_order=sorted(v), dtype=np.bool_)
    m, n = H.shape
    
    # Compute HGP code from classical code
    Hx = sp.hstack([sp.kron(sp.eye_array(m), H.T), # Im x H2
                    sp.kron(H, sp.eye_array(n)),   # H1 x In  
                   ], dtype=np.bool_).tocsc() # [ Im x H2 | H1 x In ] } m*n rows

    Hz = sp.hstack([sp.kron(H.T, sp.eye_array(m)), # H1'x Im
                    sp.kron(sp.eye_array(n), H),   # In x H2'
                   ], dtype=np.bool_).tocsc() # [ H1'x Im | In x H2'] } m*n rows
    
    H = sp.vstack([Hx, Hz])

    N = m*m + n*n
    K = N - m4ri.rank(Hx.todense()) - m4ri.rank(Hz.todense())

    # Return stacked CSS matrices H = [Hx \\ Hz], N = number of qubits and K = code dimension
    return H, N, K

def MC_erasure_plog_fixed_p(num_trials: int, state: nx.MultiGraph, p: float, 
                    rank_method: bool=False, only_X: bool=False) -> float:
    return MC_erasure_plog(num_trials, state, [p], rank_method, only_X)['mean'][0]

def MC_erasure_plog(num_trials: int, state: nx.MultiGraph, p_vals: list[float], 
                    rank_method: bool=False, only_X: bool=False) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    shape = (len(c), len(v))
    edgelist = list(nx.Graph(state).edges(data=False))
    return m4ri.MC_erasure_plog(shape, edgelist, num_trials, p_vals, rank_method, only_X)

def peel(erasure: np.array, H: sp.csr_array) -> bool:
    while np.count_nonzero(erasure):
        erased_cols, *_ = np.nonzero(erasure)
        H_E = H[:, erased_cols]

        check_degrees = np.diff(H_E.indptr)
        dangling_checks, *_ = np.nonzero(check_degrees == 1)
        
        if len(dangling_checks) == 0:
            return False
            
        dangling_check = npr.choice(dangling_checks)
        dangling_bit = erased_cols[H_E.indices[H_E.indptr[dangling_check]]]
        erasure[dangling_bit] = 0
    
    return True

def MC_peeling_classic(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    N = len(v)

    results = {'mean': [], 'std': []}
    for erasure_rate in tqdm(p_vals):
        failures = 0
        for _ in range(num_trials):
            erasure = npr.rand(N) < erasure_rate
            if not peel(erasure, H):
                failures += 1
        
        mean, std = failures/num_trials, ((failures*(num_trials - failures)) / (num_trials*(num_trials - 1)))**.5
        results['mean'].append(mean)
        results['std'].append(std)

    results['mean'] = np.array(results['mean'])
    results['std'] = np.array(results['std'])
    
    return results


def HGP(H1: sp.csr_array, H2: sp.csr_array=None):
    # Convention: H1 is the vertical axis, H2 is the horizontal axis
    # BB | BC (Z stab)
    # CB | CC
    # (X stab)
    if H2 is None:
        H2 = H1
    H1 = H1.astype(np.uint)
    H2 = H2.astype(np.uint)
    (m1, n1), (m2, n2) = H1.shape, H2.shape
    I = lambda n: sp.eye_array(n, dtype=np.uint)
    Hz = sp.hstack([sp.kron(I(n1), H2), sp.kron(H1.T, I(m2))]).asformat('csr')
    Hx = sp.hstack([sp.kron(H1, I(n2)), sp.kron(I(m1), H2.T)]).asformat('csr')
    return Hx, Hz


def HGP_peel(erasure: np.array, Hx: sp.csr_array, Hz: sp.csr_array=None, only_X=True) -> tuple[bool, np.array]:
    if Hz is None:
        assert only_X

    peel_Z = peel(erasure.copy(), Hx)
    
    if only_X:
        return peel_Z
    else:
        peel_X = peel(erasure.copy(), Hz)
        return peel_Z and peel_X
    

def MC_peeling_HGP(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    Hx, Hz = HGP(H)
    N = len(c)**2 + len(v)**2

    results = {'mean': [], 'std': []}
    for erasure_rate in tqdm(p_vals):
        failures = 0
        for _ in range(num_trials):
            erasure = npr.rand(N) < erasure_rate
            if not HGP_peel(erasure, Hx, Hz, only_X=False):
                failures += 1
        
        mean, std = failures/num_trials, ((failures*(num_trials - failures)) / (num_trials*(num_trials - 1)))**.5
        results['mean'].append(mean)
        results['std'].append(std)

    results['mean'] = np.array(results['mean'])
    results['std'] = np.array(results['std'])
    
    return results