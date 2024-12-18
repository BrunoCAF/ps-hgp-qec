from experiments_settings import codes, from_edgelist
from css_code_eval import MC_erasure_plog
import h5py
import argparse
import numpy as np

grpname = codes
p_vals = np.linspace(0.1, 0.5, 15)
MC_budget = [int(5e5), int(1e5), int(1e5), int(5e4)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    args = parser.parse_args()

    # Choose the code family
    C = args.C

    fn_data = {}

    with h5py.File('exploration.hdf5', 'r') as f:
        theta_edgelist, _ = min(((s, v) for s, v in zip(f[grpname[C]]['states'], f[grpname[C]]['values'])), key=lambda x: x[1])
        theta = from_edgelist(theta_edgelist)

    results = MC_erasure_plog(num_trials=MC_budget[C], 
                              state=theta, 
                              p_vals=p_vals)

    with h5py.File("best_from_exploration.hdf5", "a") as f: 
        grp = f.require_group(grpname[C])
        grp.create_dataset("theta", data=theta)
        grp.create_dataset("mean", data=results['mean'])
        grp.create_dataset("std", data=results['std'])
    