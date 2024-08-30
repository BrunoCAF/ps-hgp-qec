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
from experiments_settings import MC_budget, noise_levels

def arctan_diff_schedule(t: int, coef: float=10.) -> float:
    return 1./(1 + coef*t**2)

def simulated_annealing(cost_function: Callable, random_neighbor: Callable, 
                        schedule: Callable, theta0: nx.MultiGraph, epsilon: float, 
                        max_iterations: int, noisy_mode: bool=False) -> tuple[list[nx.MultiGraph], list[float]]:
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
    if noisy_mode:
        stats = cost_function(theta)
        cost, std = stats['mean'][0], stats['std'][0]
    else:
        cost = cost_function(theta)

    history, cost_history = [theta], [cost]
    delta_history, temp_history = [], []
    
    if noisy_mode:
        std_history = [std]
    

    for num_iterations in tqdm(range(max_iterations)):
        if noisy_mode and (cost + std < epsilon):
            break
        if (not noisy_mode) and (cost < epsilon):
            break
        
        temperature = schedule(num_iterations/max_iterations)

        neighbor = random_neighbor(theta)
        if noisy_mode:
            neigh_stats = cost_function(neighbor)
            neigh_cost = neigh_stats['mean'][0]
            neigh_std = neigh_stats['std'][0]
        else:
            neigh_cost = cost_function(neighbor)
        
        delta_log_cost = np.log(neigh_cost) - np.log(cost)

        delta_history.append(delta_log_cost)
        temp_history.append(temperature)
        if npr.rand() < np.exp(-delta_log_cost/temperature):
            theta = neighbor
            cost = neigh_cost
            if noisy_mode:
                std = neigh_std
    
        history.append(theta)
        cost_history.append(cost)
        if noisy_mode:
            std_history.append(std)

    if noisy_mode:
        ret = (history, cost_history, std_history, delta_history, temp_history)
    else:
        ret = (history, cost_history, delta_history, temp_history)

    return ret


sim_ann_params = {'eps': [80e-4, 5e-4, 15e-4, 250e-4], 
                  'max_iter': [2400, 900, 450, 180], 
                  'beta': [1, 4, 7, 10]}

if __name__ == '__main__':
    # Parse args: -C (Code family to optimize), -L (Length of the optimization i.e. max_iterations), 
    # -p (noise level for the cost function), -b (beta coefficient for the temperature scheduling).  
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-L', action="store", dest='max_iter', default=None, type=int)
    parser.add_argument('-p', action="store", dest='p', default=None, type=float)
    parser.add_argument('-b', action="store", dest='b', default=0, type=int, required=True)
    args = parser.parse_args()

    # Choose the code family
    C = args.C
    p = noise_levels[C] if args.p is None else args.p

    # The code family already defines some preferred values for max_iterations, epsilon, theta0
    epsilon = sim_ann_params['eps'][C]
    max_iter = sim_ann_params['max_iter'][C] if args.max_iter is None else args.max_iter
    theta0 = load_tanner_graph(path_to_initial_codes+textfiles[C])

    # Choose the temperature scheduling beta coefficient
    beta = sim_ann_params['beta'][args.b]
    
    # Define cost and scheduling functions
    cost_fn = lambda s: MC_erasure_plog(MC_budget, s, [p]) # notice that this version returns both mean and std
    sched_fn = lambda t: arctan_diff_schedule(t, coef=beta)

    # Run Simulated Annealing
    sim_ann_res = simulated_annealing(cost_function=cost_fn, random_neighbor=generate_neighbor, 
                                      schedule=sched_fn, theta0=theta0, epsilon=epsilon, 
                                      max_iterations=max_iter, noisy_mode=True)

    # Unwrap results
    theta_hist, cost_hist, std_hist, delta_hist, temp_hist = sim_ann_res

    thetas = np.row_stack([parse_edgelist(theta) for theta in theta_hist])
    costs = np.row_stack(cost_hist)
    stds = np.row_stack(std_hist)
    deltas = np.row_stack(delta_hist)
    temps = np.row_stack(temp_hist)
    
    # Store results in HDF5 file
    with h5py.File("sim_ann_deltalog.hdf5", "a") as f: 
        grp = f.require_group(codes[C])
        subgrp = grp.create_group(f'{beta=:.0f}')
        subgrp.create_dataset("theta", data=thetas)
        subgrp.create_dataset("cost", data=costs)
        subgrp.create_dataset("std", data=stds)
        subgrp.create_dataset("delta", data=deltas)
        subgrp.create_dataset("temp", data=temps)
        