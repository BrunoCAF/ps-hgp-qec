import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt
import networkx.algorithms.isomorphism as iso

from state_indexer import RewardCache

class QECEnv:
    def __init__(self, tanner_graph_params: tuple[int, int, int], 
                 plog_threshold: float):
        # Start by choosing the degrees (d_m, d_n) = e.g. (4, 3)
        # The number of edges E is a common multiple of both. 
        # Finally, m*d_m = n*d_n = E, defining (m, n). 
        self.m, self.n, self.E = tanner_graph_params
        self.d_m, self.d_n = self.E//self.m, self.E//self.n
        
        self.state = self.reset()
        self.reward_cache = RewardCache(graph_dims=tanner_graph_params[:2], 
                                        serialization='sparse')

        self.plog_threshold = plog_threshold

    def reset(self):
        return bpt.configuration_model(aseq=[self.d_m]*self.m, 
                                       bseq=[self.d_n]*self.n, seed=0, 
                                       create_using=nx.MultiGraph)

    def move(self, action: tuple[int, int]):
        # The action is described by two indices 0 <= i < j < E, where E is the number of 
        # (multi)edges, assumed constant, indicating the pair of edges to be cross-wired. 
        self.apply_action(action)

        # Evaluate the new state to issue reward. 
        if self.state not in self.reward_cache:
            plog_hat = self.estimate_plog() # The heavy lifting is done here. 
            self.reward_cache[self.state] = plog_hat
        else:
            plog_hat = self.reward_cache[self.state]
            
        reward = self.compute_reward(plog_hat)

        return self.state, reward, self.done(reward)

    def apply_action(self, edge_pair: tuple[int, int]):
        # cross-wire the two edges, i.e., update the state
        i, j = edge_pair
        edge_list = [(u, v) for u, v, _ in self.state.edges]
        e1, e2 = edge_list[i], edge_list[j]
        (c1, n1), (c2, n2) = e1, e2
        f1, f2 = (c1, n2), (c2, n1)
        self.state.remove_edges_from([e1, e2])
        self.state.add_edges_from([f1, f2])


    def estimate_plog(self) -> float:
        # this is where things get costly
        # decode the HGP code defined by the current state
        pass

    def compute_reward(self, plog_hat: float) -> float:
        # Sparse 0-1 threshold-based reward. 
        return float(plog_hat < self.plog_threshold)

    def done(self, reward: float) -> bool:
        # The episode is terminated as soon as the code achieves the desired Plog
        return reward > 0
