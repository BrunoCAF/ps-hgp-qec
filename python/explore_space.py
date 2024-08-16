import numpy as np
import scipy.sparse as sp

import networkx as nx
import networkx.algorithms.bipartite as bpt

import argparse
from tqdm import tqdm
import h5py

from simulated_annealing import generate_neighbor
from css_code_eval import MC_erasure_plog_fixed_p

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

exploration_params = [(24, 120), (15, 70), (12, 40), (8, 30)]

def load_tanner_graph(filename):
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

def parse_edgelist(state):
    return np.array(sorted(state.edges(data=False)), dtype=np.uint8).flatten() # shape: (2*E,)


if __name__ == '__main__':
    # Parse args: basically just a flag indicating the code family to explore. 
    # Optionally: args for the noise level to choose the cost function, 
    # the number of neighbors to explore, the length of the random walk. 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-N', action="store", dest='N', default=None, type=int)
    parser.add_argument('-L', action="store", dest='L', default=None, type=int)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float)
    args = parser.parse_args()

    C = args.C
    N, L = exploration_params[C] if (args.N is None or args.L is None) else (args.N, args.L)
    p = noise_levels[C] if args.p is None else args.p
    # print(f"{C = }, {N = }, {L = }, {p = }")
    
    # ------------------------------------------------------------------------------------
    states, values = [], []
    cost_fn = lambda s: MC_erasure_plog_fixed_p(MC_budget, s, p)

    # Initialize the rw with the corresponding initial state. 
    state = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # RW loop:
    for l in tqdm(range(L)):
        if l > 0:
            state = generate_neighbor(state)
        value = cost_fn(state)
        
        states.append(parse_edgelist(state))
        values.append(value)

        # Neighborhood exploration
        for n in tqdm(range(N-1)):
            neighbor = generate_neighbor(state)
            value = cost_fn(neighbor)
        
            states.append(parse_edgelist(state))
            values.append(value)

    # Exploration finished: store results in hdf5 file
    states = np.row_stack(states, dtype=np.uint8)
    values = np.row_stack(values, dtype=np.float64)
    
    with h5py.File("exploration.hdf5", "a") as f: 
        grp = f.create_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.create_dataset("states", data=states)
        grp.create_dataset("values", data=values)
