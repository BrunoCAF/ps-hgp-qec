import numpy as np
import numpy.random as npr
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp

import pym4ri as m4ri
import numba
from tqdm import tqdm

from experiments_settings import gosper_next, long_gosper_next

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

def _peel(erasure: np.array, H: sp.csr_array) -> bool:
    while np.any(erasure):
        erased_cols = np.nonzero(erasure)[0]
        H_E = H[:, erasure]
        dangling_checks = np.nonzero(np.diff(H_E.indptr) == 1)[0]
        
        if len(dangling_checks) > 0:    
            erasure[erased_cols[H_E.indices[H_E.indptr[dangling_checks[0]]]]] = 0
        else:
            return False
    
    return True

# Idea to improve peeling implementation (same algo, just more efficient implementation):
# construct the erased graph structure via adjacency lists:
# iterate over all checks (rows of sparse pcm) to count their degree towards the erasure, 
# and add them to the adjacency list of each (erased) bit in their neighborhood
# and add every degree-1 check to a stack for fast retrieval;
# keep track of the number of remaining erasures;
# while num_erasures > 0:
# look for a dangling check in the stack (if None -> failure immediately), 
# if any, pick it, find its dangling bit (only remaining erased bit in its adj list),
# unerase that bit (decrease num erasures), and while at it, decrease the degree of 
# all the checks in the adj list of this bit, also while at it, add any new dangling 
# check to the stack. 
# 

@numba.njit
def _fast_peel(erasure: np.ndarray, Hindptr: np.ndarray, Hindices: np.ndarray) -> bool:
    # Build adjacency list structure for erased bits, and keep track of their count
    erased_bits = np.nonzero(erasure)[0]
    num_erased_bits = 0
    reversed_erased_bits = np.zeros((len(erasure),), dtype=np.int32)
    for i, e in enumerate(erased_bits):
        reversed_erased_bits[e] = i
    bit_adj_lists = np.zeros((len(erasure), len(Hindptr)), dtype=np.int32)
    erased_bit_degrees = np.zeros((len(erasure),), dtype=np.int32)

    # Keep track of the degree of the checks towards the erasure and stack the dangling checks
    check_erasure_degrees = np.zeros((len(Hindptr)-1,), dtype=np.int32)
    dangling_check_stack, stack_size = np.zeros((len(Hindptr)-1,), dtype=np.int32), 0
    
    # Populate in the adjacency lists of the erased bits and add the first dangling checks
    for check_idx in range(len(Hindptr)):
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                num_erased_bits += 1
                check_erasure_degrees[check_idx] += 1
                rev_bit_idx = reversed_erased_bits[bit_idx]
                bit_adj_lists[rev_bit_idx][erased_bit_degrees[rev_bit_idx]] = check_idx
                erased_bit_degrees[rev_bit_idx] += 1

        if check_erasure_degrees[check_idx] == 1:
            dangling_check_stack[stack_size] = check_idx
            stack_size += 1

    # Main peeling loop
    while num_erased_bits > 0:
        # Find a dangling check
        dangling_check_idx = -1
        # Pop checks whose dangling bit has already been unerased
        while stack_size > 0 and dangling_check_idx == -1:
            dangling_check_idx = dangling_check_stack[stack_size-1]
            stack_size -= 1
            if check_erasure_degrees[dangling_check_idx] == 0:
                dangling_check_idx = -1

        # If no dangling checks remain, failure
        if dangling_check_idx == -1:
            return False

        # Find the corresponding dangling bit
        for bit_idx in Hindices[Hindptr[check_idx]:Hindptr[check_idx+1]]:
            if erasure[bit_idx]:
                dangling_bit_idx = bit_idx
                break
        # Unerase it and decrease the erasure count
        erasure[dangling_bit_idx] = False
        num_erased_bits -= 1
        # Decrease the degrees of each neighboring check
        rev_bit_idx = reversed_erased_bits[dangling_bit_idx]
        for neighbour_count in range(erased_bit_degrees[rev_bit_idx]):
            check_idx = bit_adj_lists[rev_bit_idx][neighbour_count]
            check_erasure_degrees[check_idx] -= 1
            # If a check becomes dangling, add it to the stack
            if check_erasure_degrees[check_idx] == 1:
                dangling_check_stack[stack_size] = check_idx
                stack_size += 1

    # If there are no erasures left, success
    return True

# @numba.njit
# def _peel(erasure: np.ndarray, H: np.ndarray) -> bool:
#     while np.any(erasure):
#         erased_cols = np.nonzero(erasure)[0]
#         H_E = H[:, erased_cols]

#         check_degrees = np.sum(H_E, axis=1)
#         dangling_checks = np.nonzero(check_degrees == 1)[0]

#         if len(dangling_checks) == 0:
#             return False

#         # Select a random dangling check manually
#         dangling_check = npr.choice(dangling_checks)
#         dangling_bit = erased_cols[np.argmax(H_E[dangling_check])]

#         erasure[dangling_bit] = 0

#     return True

def peel(erasure: np.array, H: sp.csr_array) -> bool:
    return _peel(erasure, H)
    # return _fast_peel(erasure, H.indptr, H.indices)


def MC_peeling_classic(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    N = len(v)

    results = {'mean': [], 'std': []}
    if len(p_vals) > 1:
        iterator = tqdm(p_vals)
    else:
        iterator = p_vals

    for erasure_rate in iterator:
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

# @numba.njit
def _HGP_peel(erasure: np.array, Hx: sp.csr_array, Hz: sp.csr_array=None, only_X:bool=True) -> bool:
    if Hz is None:
        assert only_X

    peel_X = peel(erasure.copy(), Hz)

    if only_X:
        return peel_X
    else:
        peel_Z = peel(erasure.copy(), Hx)
        return peel_X and peel_Z
    

# @numba.njit
def _MC_peeling_HGP(num_trials: int, Hx: sp.csr_array, Hz: sp.csr_array, p_vals: list[float]) -> dict[str, np.ndarray]:
    N = Hx.shape[1]

    results = {'mean': np.zeros((len(p_vals),)), 'std': np.zeros((len(p_vals),))}
    for t, erasure_rate in enumerate(p_vals):
        failures = 0
        for _ in range(num_trials):
            erasure = npr.rand(N) < erasure_rate
            if not _HGP_peel(erasure, Hx, Hz, only_X=False):
                failures += 1

        mean, std = failures/num_trials, ((failures*(num_trials - failures)) / (num_trials*(num_trials - 1)))**.5
        results['mean'][t] = mean
        results['std'][t] = std

    return results

def MC_peeling_HGP(num_trials: int, state: nx.MultiGraph, p_vals: list[float]) -> dict:
    c = [n for n, b in state.nodes(data='bipartite') if b == 0]
    v = [n for n, b in state.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(state, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    Hx, Hz = HGP(H)
    return _MC_peeling_HGP(num_trials, Hx, Hz, p_vals)

# @numba.jit
def stabilizer_search(H: np.ndarray, erasure: np.ndarray, depth: int) -> tuple[list[bool], np.ndarray]:
    m, _ = H.shape
    found_at_depth = [True]*depth
    for weight in range(1, depth+1):
        c = (1 << weight) - 1  # Smallest subset of size 'weight'
        while c < (1 << m):
            # Convert 'c' to a binary representation as a NumPy array
            row_combination = np.array([(c >> i) & 1 for i in range(m)], dtype=bool) # , dtype=np.float32
            stabilizer = np.bitwise_xor.reduce(H[row_combination, :], axis=0, dtype=bool)
            
            # Check if corresponding stabilizer lies within the erasure
            if np.any(stabilizer) and not np.any(stabilizer >> erasure):
                return found_at_depth, stabilizer

            c = long_gosper_next(c)  # Get next subset using Gosper's hack

        found_at_depth[weight-1] = False
        # print(f'Failed at depth {weight}!')
        
    return found_at_depth, np.zeros_like(erasure)