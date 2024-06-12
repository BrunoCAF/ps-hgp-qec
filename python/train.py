import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import time
from tqdm import tqdm

from agent import PSAgent
from rl_environment import QECEnv

class Interaction:
    def __init__(self, agent: PSAgent, environment: QECEnv):
        # Receive an agent and an environment objects
        self.agent = agent
        self.environment = environment

        
        self.num_episodes = 1000
        self.num_steps_per_episode = 1000

    def train(self, num_episodes: int = 1000, steps_per_episode: int = 1000) -> np.array:
        # run the interactions between agent and environment
        # store reward history, record metrics, temporize, etc
        
        learning_curve = np.zeros(num_episodes)
        reward = 0
        for episode in tqdm(range(num_episodes)):
            cumulative_reward = 0
            discretized_observation = self.env.reset()
            for t in range(steps_per_episode):
                discretized_observation, reward, done = self.single_interaction_step(discretized_observation, reward)
                cumulative_reward += reward
                if done:
                    break
                learning_curve[episode] = float(cumulative_reward)/(t+1)
        return learning_curve



if __name__ == 'main':
    # initialize an Interaction object
    # run its train method
    # record results, monitor metrics, report, etc
    pass