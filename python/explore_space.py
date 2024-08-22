import numpy as np
import scipy.sparse as sp

import networkx as nx
import networkx.algorithms.bipartite as bpt

import argparse
from tqdm import tqdm
import h5py

from css_code_eval import MC_erasure_plog
from experiments_settings import load_tanner_graph, parse_edgelist, generate_neighbor
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import MC_budget, noise_levels

exploration_params = [(24, 120), (15, 70), (12, 40), (8, 30)]


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
    states, values, stds = [], [], []
    cost_fn = lambda s: MC_erasure_plog(MC_budget, s, [p])

    # Initialize the rw with the corresponding initial state. 
    state = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # RW loop:
    for l in tqdm(range(L)):
        if l > 0:
            state = generate_neighbor(state)
        stat = cost_fn(state)
        value = stat['mean'][0]
        std = stat['std'][0]
        
        states.append(parse_edgelist(state))
        values.append(value)
        stds.append(std)

        # Neighborhood exploration
        for n in tqdm(range(N-1)):
            neighbor = generate_neighbor(state)
            stat = cost_fn(neighbor)
            value = stat['mean'][0]
            std = stat['std'][0]
        
            states.append(parse_edgelist(neighbor))
            values.append(value)
            stds.append(std)

    # Exploration finished: store results in hdf5 file
    states = np.row_stack(states, dtype=np.uint8)
    values = np.row_stack(values, dtype=np.float64)
    stds = np.row_stack(stds, dtype=np.float64)
    
    with h5py.File("exploration.hdf5", "a") as f: 
        grp = f.require_group(codes[C])
        grp.attrs['MC_budget'] = MC_budget
        grp.attrs['p'] = p
        grp.create_dataset("states", data=states)
        grp.create_dataset("values", data=values)
        grp.create_dataset("stds", data=stds)
