from experiments_settings import codes, from_parity_check_matrix
from css_code_eval import MC_erasure_plog
import h5py
import argparse
import numpy as np
import pickle

names = ["PEG_codes", "SA_codes", "PS_codes", "PE_codes"]
objs = []
for name in names:
#     with open(name+'.pkl', 'wb') as f:
#         pickle.dump(obj, f)
    with open(name+'.pkl', 'rb') as f:
        objs.append(pickle.load(f))

grpname = codes
p_vals = np.linspace(0.1, 0.5, 15)
MC_budget = [int(5e5), int(1e5), int(1e5), int(5e4)]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-F', action="store", dest='F', default=0, type=int, required=True)
    args = parser.parse_args()

    # Choose the code family, code length, error rate, MC budget
    F, C = args.F, args.C
    
    family = objs[F]
    code = list(family.keys())[C]
    
    theta = from_parity_check_matrix(family[code])

    results = MC_erasure_plog(num_trials=MC_budget[C], 
                              state=theta, 
                              p_vals=p_vals, 
                              only_X=True)

    with h5py.File("erasure_only_X.hdf5", "a") as f: 
        grp = f.require_group(names[F])
        subgrp = grp.require_group(code)
        
        subgrp.create_dataset("theta", data=theta)
        subgrp.create_dataset("mean", data=results['mean'])
        subgrp.create_dataset("std", data=results['std'])
    