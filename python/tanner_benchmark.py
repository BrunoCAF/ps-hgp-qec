import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt

import h5py
import argparse
from tqdm import tqdm
import time

from css_code_eval import HGP, MC_erasure_plog, MC_peeling_HGP
from experiments_settings import code_dimension, code_distance

ord_ass_28_4_13 = [
    np.roll(np.concatenate([np.array([0]), (1+np.roll(np.arange(7), shift=1))]), shift=k) for k in range(8)
]

ord_ass_28_6_10 = [
    [0, 1, 2, 4, 3, 6, 7, 5], 
    [4, 0, 5, 6, 7, 3, 2, 1], 
    [4, 1, 0, 5, 6, 7, 3, 2], 
    [4, 2, 1, 0, 5, 6, 7, 3], 
    [4, 3, 2, 1, 0, 5, 6, 7], 
    [4, 7, 3, 2, 1, 0, 5, 6], 
    [4, 6, 7, 3, 2, 1, 0, 5], 
    [4, 5, 6, 7, 3, 2, 1, 0], 
]

ord_ass_28_10_6 = [
    [u^v for v in range(8)] for u in range(8)
]

order_assignment_list = [ord_ass_28_4_13, ord_ass_28_6_10, ord_ass_28_10_6]

def tanner_code_K8_Hamming(order_assignment: list[list[int]]) -> sp.csr_array:
    K8 = nx.complete_graph(8)
        
    indptr, indices, data = [0], [], []
    for u in K8.nodes:
        count = 0
        for i, e in enumerate(K8.edges):
            if u in e:
                count += 1
                indices.append(i)
                v = [w for w in e if w != u][0]
                x = np.unpackbits(np.uint8(order_assignment[u][v]), count=3, bitorder='little').astype(np.int32).reshape(3, 1)
                data.append(x)

        indptr.append(count)
    indptr = np.cumsum(indptr)
    indices = np.array(indices)
    data = np.stack(data, axis=0)

    return sp.csr_array(sp.bsr_array((data, indices, indptr)).todense())


p_vals = np.linspace(0.1, 0.5, 15)
MC_budget = int(1e5)

decoders = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-E', action="store", dest='E', default=0, type=int, required=True)
    parser.add_argument('-D', action="store", dest='D', default=0, type=int)
    parser.add_argument('-M', action="store", dest='MC', default=MC_budget, type=int)
    args = parser.parse_args()

    # Choose the code, error rate, decoder and MC budget
    C, E, D = args.C, args.E, args.D
    MC_budget = args.MC
    MC_ML = int(1e6) if E < 3 else MC_budget
    MC_peel = MC_budget if E < 3 else int(1e3)
    print(f'Script configuration: {C = }, {E = }, {D = }, {MC_ML = }, {MC_peel = }')

    # Set code
    code = tanner_code_K8_Hamming(order_assignment_list[C])
    n, k, d = code.shape[1], code_dimension(code), code_distance(code)
    nt, kt, dt = code.shape[0], code_dimension(code.T), code_distance(code.T)
    print(f'Classical (base) code params: [n={n}, k={k}, d={d}], [n^t={nt}, k^t={kt}, d^t={dt}]')
    
    print(f'Quantum HGP code params: [[N={n*n+nt*nt}, K={k*k+kt*kt}, D={min(d, dt)}]]')
    params = {
        'n': n, 'nt': nt, 'k': k, 
        'kt': kt, 'd': d, 'dt': dt, 
        'N': n*n+nt*nt, 'K': k*k+kt*kt, 'D': min(d, dt), 
    }

    # Set simulation parameters
    er = p_vals[E]
    print(f'Error rate: {er:.3f} | MC budget (ML): 10^{np.log10(MC_ML):.0f} trials; (peel): 10^{np.log10(MC_peel):.0f} trials')

    # Set decoder/cost function -> do both ML and peeling
    theta = bpt.from_biadjacency_matrix(code, create_using=nx.MultiGraph)
    print('Running ML decoding benchmark...')    
    ML_results = MC_erasure_plog(MC_ML, state=theta, p_vals=[er])
    print('Running peeling decoding benchmark...')    
    peeling_results = MC_peeling_HGP(MC_peel, state=theta, p_vals=[er])

    # Save results
    print('Simulation done, saving results...')    
    time.sleep(E)
    with h5py.File("tanner_codes_benchmark.hdf5", "a") as f: 
        grp = f.require_group(f'[{n},{k},{d}]')
        
        for par, val in params.items():
            grp.attrs[str(par)] = val

        subgrp = grp.require_group(f'ER={E}')
        
        subgrp.create_dataset("ML_ler", data=ML_results['mean'])
        subgrp.create_dataset("ML_eb", data=1.96*ML_results['std']/np.sqrt(MC_ML))
        subgrp.create_dataset("peel_ler", data=peeling_results['mean'])
        subgrp.create_dataset("peel_eb", data=1.96*peeling_results['std']/np.sqrt(MC_peel))
    
    print('Results saved. All done.')