import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt

from state_indexer import RewardCache
from css_code_eval import MC_erasure_plog_fixed_p

class QECEnv:
    def __init__(self, initial_state: nx.MultiGraph, plog_threshold: float, 
                 MC_budget: int=int(1e4), erasure_rate: float=0.3):
        self.initial_state = initial_state
        diam = lambda arr: np.max(arr) - np.min(arr) + 1
        edges = np.array(sorted(initial_state.edges(data=False)))
        self.m, self.n = np.apply_along_axis(diam, 0, edges)
        self.E = len(edges)
    
        self.state = self.reset()
        self.reward_cache = RewardCache(graph_dims=(self.m, self.n), 
                                        serialization='sparse')

        self.plog_threshold = plog_threshold
        self.MC_budget = MC_budget
        self.erasure_rate = erasure_rate

        self.plog_hat = self.estimate_plog()
        self.reward_cache[self.state] = self.plog_hat

    def reset(self):
        return nx.MultiGraph(self.initial_state)

    def move(self, action: tuple[int, int]):
        # The action is described by two indices 0 <= i < j < E, where E is the number of 
        # (multi)edges, assumed constant, indicating the pair of edges to be cross-wired. 
        self.apply_action(action)

        # Evaluate the new state to issue reward. 
        if self.state not in self.reward_cache:
            self.plog_hat = self.estimate_plog() # The heavy lifting is done here. 
            self.reward_cache[self.state] = self.plog_hat
        else:
            self.plog_hat = self.reward_cache[self.state]
            
        reward = self.compute_reward(self.plog_hat)

        return self.state, reward, self.done(reward)

    def apply_action(self, edge_pair: tuple[int, int]):
        # cross-wire the two edges, i.e., update the state
        i, j = edge_pair
        assert 0 <= i < j < self.E
        edge_list = sorted(self.state.edges(data=False))
        e1, e2 = edge_list[i], edge_list[j]
        (c1, n1), (c2, n2) = e1, e2
        f1, f2 = (c1, n2), (c2, n1)
        self.state.remove_edges_from([e1, e2])
        self.state.add_edges_from([f1, f2])


    def estimate_plog(self) -> float:
        # Decode the HGP code defined by the current state via Monte Carlo simulation. 
        return MC_erasure_plog_fixed_p(self.MC_budget, self.state, self.erasure_rate)

    def compute_reward(self, plog_hat: float) -> float:
        # Sparse 0-1 threshold-based reward. 
        # return float(plog_hat < self.plog_threshold)
        return np.maximum(np.floor(np.log(self.plog_threshold / plog_hat)), 0)

    def done(self, reward: float) -> bool:
        # The episode is terminated as soon as the code achieves the desired Plog
        # return reward > 0
        # No early termination
        return False
