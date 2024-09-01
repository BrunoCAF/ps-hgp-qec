import numpy as np
import numpy.random as npr
import scipy.sparse as sp
from scipy.special import comb
import networkx as nx

from state_indexer import StateIndexer

class PSAgent:
    def __init__(self, state_space_params: tuple, action_space_params: str='cross-wiring', 
                 beta:float=1e-1, gamma:float=1e-1, eta:float=1e-1):
        self.state_space_params = state_space_params
        self.action_space_params = action_space_params

        self.state_indexer = StateIndexer(self.state_space_params[:2], serialization='sparse')
        self.encode_action = {'cross-wiring': self.cross_wiring}[self.action_space_params]

        self.beta = beta   # Softmax beta parameter
        self.gamma = gamma # Forgetting/h damping gamma
        self.eta = eta     # Glow damping eta parameter

        # The state space params consist of the numbers of nodes and 
        # edges of the Tanner graph defining the classical codes
        m, n, E = self.state_space_params
        self.S, self.A = min(1<<8, comb(m*n, E, exact=True)), comb(E, 2, exact=True)
        # CSR format is good for arithmetic and row-slicing (update weights and sample actions)
        self.h_matrix = sp.csr_array((self.S, self.A), dtype=np.float64)
        # DOK format is good for entry-wise incremental construction of the matrix
        self.g_matrix = sp.dok_array((self.S, self.A), dtype=np.float64)

    def learn_and_act(self, observation: nx.MultiGraph, reward: float) -> tuple[int, int]:
        # Update h matrix
        self.h_matrix *= 1 - self.gamma
        self.h_matrix += reward * self.g_matrix
        
        # Sample action from current state's distribution
        s = self.preprocess_percept(observation)
        row = self.h_matrix[[s]]
        probs = np.exp(self.beta * row.data) - 1
        q = probs.sum()/(probs.sum() + row.shape[1])
        if npr.rand() < q:
            probs /= probs.sum()
            a = npr.choice(row.indices, p=probs)
        else:
            a = npr.randint(row.shape[1])

        # Update glow matrix
        self.g_matrix *= 1 - self.eta
        self.g_matrix[s, a] = 1
        
        # Return encoded action
        return self.encode_action(a)

    def preprocess_percept(self, observation: nx.MultiGraph) -> int:
        if self.state_indexer.next_index >= self.S:
            self.S <<= 1
            self.h_matrix.resize((self.S, self.A))
            self.g_matrix.resize((self.S, self.A))

        return self.state_indexer.get_index(observation)

    def cross_wiring(self, a: int) -> tuple[int, int]:
        # a is the index of the action in the range [0..A-1], convert it 
        # to a pair of indices representing the edges to be cross-wired. 
        E = self.state_space_params[2]
        i = np.floor(((2*E - 1) - np.sqrt((2*E-1)**2 - 8*a))//2).astype(int)
        j = (a - E*i + ((i+2)*(i+1))//2)
        return i, j