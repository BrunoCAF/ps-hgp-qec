import numpy as np
import numpy.random as npr

import networkx as nx

import argparse
import h5py

from css_code_eval import MC_peeling_classic, MC_peeling_HGP
from experiments_settings import load_tanner_graph, parse_edgelist, generate_neighbor
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import p_vals

from simulated_annealing import arctan_diff_schedule, simulated_annealing

sim_ann_params = {'max_iter': [800, 500, 400, 300], 
                  'beta': 10}

MC_budget = int(1e4)
noise_levels = [0.3, 0.3, 0.35, 0.35]

if __name__ == '__main__':
    # Parse args: -C (Code family to optimize), 
    # -L (Length of the optimization i.e. max_iterations), 
    # -p (noise level for the cost function) 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-m', action="store", dest='mode', default=0, type=int)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    p = noise_levels[C]

    # Choose the evaluation criterion/mode (classic vs HGP)
    mode = args.m
    eval_fn = [MC_peeling_classic, MC_peeling_HGP][mode]

    # The code family already defines some preferred values for max_iterations, theta0
    max_iter = sim_ann_params['max_iter'][C]
    if mode == 1:
        max_iter //= 10
    theta0 = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # Define cost and scheduling functions
    # notice that this version returns both mean and std
    cost_fn = lambda s: eval_fn(MC_budget, s, [p]) 
    sched_fn = lambda t: arctan_diff_schedule(t, coef=sim_ann_params['beta'])

    # Add verbose to help debugging
    print(f"{C = }, {p = :.3f}, {max_iter = }")
    print("Parameters all set, starting the optimization now")

    # Run Simulated Annealing
    sim_ann_res = simulated_annealing(cost_function=cost_fn, random_neighbor=generate_neighbor, 
                                      schedule=sched_fn, theta0=theta0, max_iterations=max_iter)

    # Add verbose to help debugging
    print("Optimization routine finished. Collecting results")

    # Unwrap results
    theta_hist, cost_hist, std_hist, best_theta, best_cost, best_std = sim_ann_res

    thetas = np.row_stack([parse_edgelist(theta) for theta in theta_hist])
    costs, stds = np.row_stack(cost_hist), np.row_stack(std_hist)
    
    # Add verbose to help debugging
    print("Saving results")
    
    # Store results in HDF5 file
    with h5py.File("SA_peeling.hdf5", "a") as f: # TODO: include the stopping sets
        grp = f.require_group(codes[C])
        grp.create_dataset("theta", data=thetas)
        grp.create_dataset("cost", data=costs)
        grp.create_dataset("std", data=stds)

    # Add verbose to help debugging
    print("Results saved. Now we draw the curves for the best codes found during the simulation")

    # Analyze the best code found during the simulation (and the initial one)
    classic_results_0 = MC_peeling_classic(num_trials=10*MC_budget, state=theta0, p_vals=p_vals)
    classic_results_best = MC_peeling_classic(num_trials=10*MC_budget, state=best_theta, p_vals=p_vals)
    
    HGP_results_0 = MC_peeling_HGP(num_trials=MC_budget, state=theta0, p_vals=p_vals)
    HGP_results_best = MC_peeling_HGP(num_trials=MC_budget, state=best_theta, p_vals=p_vals)
    # TODO: include the stopping sets

    # Add verbose to help debugging
    print("Saving curve results.")

    with h5py.File("best_peeling.hdf5", "a") as f: 
        grp = f.require_group(codes[C])
        
        initial_grp = grp.require_group("initial")
        initial_grp.create_dataset("theta", data=parse_edgelist(theta0))
        initial_grp.create_dataset("mean_peel_classic", data=classic_results_0['mean'])
        initial_grp.create_dataset("std_peel_classic", data=classic_results_0['std'])
        initial_grp.create_dataset("mean_peel_hgp", data=HGP_results_0['mean'])
        initial_grp.create_dataset("std_peel_hgp", data=HGP_results_0['std'])

        best_grp = grp.require_group("best")
        best_grp.create_dataset("theta", data=parse_edgelist(best_theta))
        best_grp.create_dataset("mean_peel_classic", data=classic_results_best['mean'])
        best_grp.create_dataset("std_peel_classic", data=classic_results_best['std'])
        best_grp.create_dataset("mean_peel_hgp", data=HGP_results_best['mean'])
        best_grp.create_dataset("std_peel_hgp", data=HGP_results_best['std'])

    # Add verbose to help debugging
    print("All done.")
        