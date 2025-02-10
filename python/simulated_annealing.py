import numpy as np
import numpy.random as npr

import networkx as nx

from typing import Callable
import argparse
from tqdm import tqdm
import h5py

from css_code_eval import MC_erasure_plog
from experiments_settings import load_tanner_graph, parse_edgelist, generate_neighbor
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import MC_budget, noise_levels, p_vals

def arctan_diff_schedule(t: int, coef: float=10.) -> float:
    return 1./(1 + coef*t**2)

def simulated_annealing(cost_function: Callable, random_neighbor: Callable, schedule: Callable,
                        theta0: nx.MultiGraph, max_iterations: int, epsilon: float=0.0) -> tuple:
    """
    Executes the Simulated Annealing (SA) algorithm to minimize (optimize) a cost function 
    over the state space of Tanner graphs with fixed vertex sets and number of (multi)edges. 

    :param cost_function: function to be minimized.
    :param random_neighbor: function which returns a random neighbor of a given point.
    :param schedule: function which computes the temperature schedule.
    :param theta0: initial state.
    :param epsilon: used to stop the optimization if the current cost is less than epsilon.
    :param max_iterations: maximum number of iterations.
    :return history: history of points visited by the algorithm.
    :return cost_history: cost function values along the history.
    """
    theta = theta0
    
    stats = cost_function(theta)
    cost, std = stats['mean'][0], stats['std'][0]
    best_theta, best_cost, best_std = theta, cost, std

    history, cost_history, std_history = [theta], [cost], [std]
    
    for num_iterations in tqdm(range(max_iterations)):
        if cost < epsilon:
            break
        
        temperature = schedule(num_iterations/max_iterations)

        neighbor = random_neighbor(theta)

        neigh_stats = cost_function(neighbor)
        neigh_cost, neigh_std = neigh_stats['mean'][0], neigh_stats['std'][0]
        
        delta_log_cost = np.log(neigh_cost) - np.log(cost)

        if npr.rand() < np.exp(-delta_log_cost/temperature):
            theta = neighbor
            cost, std = neigh_cost, neigh_std

            best_theta, best_cost, best_std = min([(theta, cost, std), (best_theta, best_cost, best_std)], key=lambda p: p[1])
    
        history.append(theta)
        cost_history.append(cost)
        std_history.append(std)

    return (history, cost_history, std_history, best_theta, best_cost, best_std)


sim_ann_params = {'max_iter': [2400, 900, 450, 180], 
                  'beta': 10}

if __name__ == '__main__':
    # Parse args: -C (Code family to optimize), -L (Length of the optimization i.e. max_iterations), 
    # -p (noise level for the cost function) 
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-L', action="store", dest='max_iter', default=None, type=int)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    p = noise_levels[C] if args.p is None else args.p

    # The code family already defines some preferred values for max_iterations, theta0
    max_iter = sim_ann_params['max_iter'][C] if args.max_iter is None else args.max_iter
    theta0 = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # Define cost and scheduling functions
    cost_fn = lambda s: MC_erasure_plog(MC_budget, s, [p]) # notice that this version returns both mean and std
    sched_fn = lambda t: arctan_diff_schedule(t, coef=sim_ann_params['beta'])

    # Run Simulated Annealing
    sim_ann_res = simulated_annealing(cost_function=cost_fn, random_neighbor=generate_neighbor, 
                                      schedule=sched_fn, theta0=theta0, max_iterations=max_iter)

    # Unwrap results
    theta_hist, cost_hist, std_hist, best_theta, best_cost, best_std = sim_ann_res

    thetas = np.row_stack([parse_edgelist(theta) for theta in theta_hist])
    costs, stds = np.row_stack(cost_hist), np.row_stack(std_hist)
    
    # Store results in HDF5 file
    with h5py.File("sim_ann_deltalog.hdf5", "a") as f: 
        grp = f.require_group(codes[C])
        grp.create_dataset("theta", data=thetas)
        grp.create_dataset("cost", data=costs)
        grp.create_dataset("std", data=stds)

    # Analyze the best code found during the simulation
    results = MC_erasure_plog(num_trials=MC_budget[C], 
                              state=best_theta, 
                              p_vals=p_vals)

    with h5py.File("best_from_sadl.hdf5", "a") as f: 
        grp = f.require_group(codes[C])
        grp.create_dataset("mean", data=results['mean'])
        grp.create_dataset("std", data=results['std'])
        