from experiments_settings import codes, from_edgelist
from css_code_eval import MC_erasure_plog
import h5py
import argparse
import numpy as np

grpname = codes
subgrpname = [f"{beta=:.0f}" for beta in [1, 4, 7, 10]] + ['initial']
p_vals = np.linspace(0.1, 0.5, 15)
MC_budget = [int(5e5), int(1e5), int(1e5), int(5e4)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-b', action="store", dest='b', default=0, type=int, required=True)
    parser.add_argument('-M', action="store", dest='MC', default=None, type=int)
    parser.add_argument('-p', action="store", dest='p', default=None, type=tuple[float])
    args = parser.parse_args()

    # Choose the code family and the beta parameter from which to pick the best (beta=0 for initial code)
    C, b = args.C, args.b

    with h5py.File("best_from_sadl.hdf5", "r") as f: 
        theta = from_edgelist(f[grpname[C]][subgrpname[b]]['theta'][()])

    results = MC_erasure_plog(num_trials=MC_budget[C], 
                              state=theta, 
                              p_vals=p_vals)

    with h5py.File("best_from_sadl.hdf5", "a") as f: 
        grp = f.require_group(grpname[C])
        subgrp = grp.require_group(subgrpname[b])
        subgrp.create_dataset("mean", data=results['mean'])
        subgrp.create_dataset("std", data=results['std'])
    