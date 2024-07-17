import numpy as np
import numpy.random as npr

from scipy.special import comb
import networkx as nx

from typing import Callable
from tqdm import tqdm

def generate_neighbor(theta: nx.MultiGraph) -> nx.MultiGraph:
    # Copy state
    neighbor = nx.MultiGraph(theta)
    
    # get (multi)edge number from state theta
    E = neighbor.number_of_edges()

    # compute action space size
    A = comb(E, 2, exact=True)
    
    # sample action
    a = npr.choice(A)
    
    # convert to edge indices
    i = np.floor(((2*E - 1) - np.sqrt((2*E-1)**2 - 8*a))//2).astype(int)
    j = (a - E*i + ((i+2)*(i+1))//2)
    
    # apply cross-wiring 
    edge_list = [(u, v) for u, v, _ in neighbor.edges]
    e1, e2 = edge_list[i], edge_list[j]
    (c1, n1), (c2, n2) = e1, e2
    f1, f2 = (c1, n2), (c2, n1)
    neighbor.remove_edges_from([e1, e2])
    neighbor.add_edges_from([f1, f2])
    
    return neighbor

def arctan_diff_schedule(t: int) -> float:
    coef = 1.
    return 1./(1 + coef*t**2)

def simulated_annealing(cost_function: Callable, random_neighbor: Callable, 
                        schedule: Callable, theta0: nx.MultiGraph, epsilon: float, 
                        max_iterations: int) -> tuple[list[nx.MultiGraph], list[float]]:
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
    cost = cost_function(theta)
    history, cost_history = [theta], [cost]
    
    for num_iterations in tqdm(range(max_iterations)):
        if cost < epsilon:
            break
        
        temperature = schedule(num_iterations)

        neighbor = random_neighbor(theta)
        neigh_cost = cost_function(neighbor)
        delta_e = neigh_cost - cost

        if npr.rand() < np.exp(-delta_e/temperature):
            theta = neighbor
            cost = neigh_cost
    
        history.append(theta)
        cost_history.append(cost)

    return history, cost_history