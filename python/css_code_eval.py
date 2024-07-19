import numpy as np
import numpy.random as npr
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp

import pym4ri as m4ri

# TODO: implement some strategy for importance sampling

def CSS_HGP_code_from_state(state: nx.MultiGraph) -> tuple[sp.sparray, int, int]:
    # Convert multigraph state to simple graph
    G = nx.Graph(state)
    
    # Extract biadjacency matrix from the Tanner graph of the classical code
    c, v = bpt.sets(G)
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
    c, v = bpt.sets(state)
    shape = (len(c), len(v))
    edgelist = list(nx.Graph(state).edges(data=False))
    return m4ri.MC_erasure_plog(shape, edgelist, num_trials, p_vals.tolist(), rank_method, only_X)
