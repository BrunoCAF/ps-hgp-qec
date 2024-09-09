import numpy as np
import scipy.sparse as sp
import networkx as nx

import argparse
from tqdm import tqdm
import h5py

from agent import PSAgent
from rl_environment import QECEnv

from experiments_settings import load_tanner_graph, parse_edgelist
from experiments_settings import codes, path_to_initial_codes, textfiles
from experiments_settings import MC_budget, noise_levels

class Interaction:
    def __init__(self, agent: PSAgent, environment: QECEnv, storage_path: str):
        self.agent = agent
        self.environment = environment

        self.storage_path = storage_path

        self.state_history = []
        self.cost_history = []
        self.episode_lengths = np.array([], dtype=int)


    def store(self):
        # Assume training was completed, history and results were recorded
        print('Stacking history...')
        self.state_history = np.row_stack(self.state_history)
        self.cost_history = np.row_stack(self.cost_history)

        # Create inverted index for states
        print('Creating inverted state index...')
        inv_idx_dict = self.agent.state_indexer.invert_index()
        inv_idx_list = [np.frombuffer(inv_idx_dict[i], dtype=np.uint8) for i in range(len(inv_idx_dict))]
        inv_idx_arr = np.row_stack(inv_idx_list)

        print('Opening hdf5 file...')
        with h5py.File('ps_train_noterm_logrew.hdf5', 'a') as f:
            # storage path is of the form: code/hard|easy/beg=xyz/dsets+attrs
            grp = f.require_group(self.storage_path)
            
            # Save agent and environment parameters
            print('Creating attributes...')
            grp.attrs['beta'] = self.agent.beta
            grp.attrs['eta'] = self.agent.eta
            grp.attrs['gamma'] = self.agent.gamma
            grp.attrs['threshold'] = self.environment.plog_threshold
            
            # Save inverted index and trained policy
            print('Saving data from agent...')
            # - inverted state indexer from agent;
            grp.create_dataset('index', data=inv_idx_arr)
            # - h and g matrices;
            grp.create_dataset('h_data', data=self.agent.h_matrix.data)
            grp.create_dataset('h_indices', data=self.agent.h_matrix.indices)
            grp.create_dataset('h_indptr', data=self.agent.h_matrix.indptr)

            grp.create_dataset('g_keys', data=np.array(list(self.agent.g_matrix.keys()), dtype='i,i'))
            grp.create_dataset('g_vals', data=np.array(list(self.agent.g_matrix.values())))

            # Save full training history and episode lengths
            print('Saving data from training...')
            # - state at each step;
            grp.create_dataset('states', data=self.state_history)
            # - plog at each step;
            grp.create_dataset('costs', data=self.cost_history)
            # - episode length;
            grp.create_dataset('ep_len', data=self.episode_lengths)

        print('Saving completed.')
        

    def train(self, num_episodes: int, steps_per_episode: int):
        # run the interactions between agent and environment
        # store reward history, record metrics, temporize, etc
        self.episode_lengths = np.zeros(num_episodes, dtype=int)
        
        reward = 0
        initial_cost = self.environment.plog_hat
        for episode in tqdm(range(num_episodes), desc='Training: '):
            state = self.environment.reset()
            self.state_history.append(parse_edgelist(state))
            self.cost_history.append(initial_cost)

            for t in tqdm(range(steps_per_episode), desc=f'Episode {episode}: '):
                action = self.agent.learn_and_act(state, reward)
                state, reward, done = self.environment.move(action)

                self.state_history.append(parse_edgelist(state))
                self.cost_history.append(self.environment.plog_hat)

                if done:
                    break
            
            self.episode_lengths[episode] = t+2
        


ps_params = {'plog_thres_hard': [2e-2, 3e-3, 4e-3, 4e-2], 
             'plog_thres_easy': [3e-2, 5e-3, 6e-3, 5e-2], 
             'num_episodes': [20, 10, 8, 5], 
             'steps_per_episode': [120, 70, 50, 35], 
             'beta': [2., 1., 0.5], 
             'eta': [1, 1/2, 1/4], 
             'gamma': [1, 3/4, 1/2]}

# Estimated time per step: [25, 60, 100, 210]
# Approx limit of evals: [2280, 760, 400, 175]

def set_agent_params(A: int, b: int, e: int, g: int, L: int) -> tuple[float, float, float]:
    beta = np.log(A * ps_params['beta'][b])
    q = np.log(A * ps_params['beta'][b] * ps_params['eta'][e])
    eta = 1 - (q / beta)**(1/L)
    r = np.log(A * ps_params['beta'][b] * ps_params['eta'][e] * ps_params['gamma'][g])
    gamma = 1 - (r / q)**(1/L)
    return beta, eta, gamma


if __name__ == '__main__':
    # Parse args: 
    # Environment params: initial code, MC_budget, erasure rate, plog threshold
    # Agent params: gamma damping, eta glow damping, softmax beta
    # -C (Code family -> initial, plog_threshold, erasure_rate, MC_budget), 
    # -t (plog_threshold level)
    parser = argparse.ArgumentParser()
    parser.add_argument('-C', action="store", dest='C', default=0, type=int, required=True)
    parser.add_argument('-t', action="store", dest='t', default=0, type=int, required=True)
    parser.add_argument('-b', action="store", dest='b', default=0, type=int, required=True)
    parser.add_argument('-e', action="store", dest='e', default=0, type=int, required=True)
    parser.add_argument('-g', action="store", dest='g', default=0, type=int, required=True)
    args = parser.parse_args()

    # Set training and environment options
    C = args.C
    t = 'plog_thres_hard' if args.t == 0 else 'plog_thres_easy'
    plog_threshold = ps_params[t][C]
    erasure_rate = noise_levels[C]
    N, L = ps_params['num_episodes'][C], ps_params['steps_per_episode'][C]

    print('Initializing environment...')
    env = QECEnv(load_tanner_graph(path_to_initial_codes+textfiles[C]), 
                 plog_threshold, MC_budget, erasure_rate)
    
    # Set agent options
    A = (env.E * (env.E - 1))//2
    beta, eta, gamma = set_agent_params(A, args.b, args.e, args.g, L)

    print('Initializing agent...')
    agent = PSAgent((env.m, env.n, env.E), 'cross-wiring', 
                    beta=beta, gamma=gamma, eta=eta)
    
    # Set path for storage of results
    storage_path = f'{codes[C]}/{t.split('_')[-1]}/beg={args.b}{args.e}{args.g}'

    print('Summary of options:')
    print(f'Code family: {codes[C]} | # of episodes: {N} | max ep length: {L}')
    print(f'MC trials: {MC_budget:.0g} | erasure rate: {erasure_rate:.2f} | P_log threshold: {plog_threshold:.0g}')
    print(f'Agent params: {beta = :.3g}, {eta = :.3g}, {gamma = :.3g}')

    # Initialize Interaction, train and save results
    interaction_obj = Interaction(agent=agent, environment=env, storage_path=storage_path)

    print('Starting training...')
    interaction_obj.train(num_episodes=N, steps_per_episode=L)

    print('Training finished. Saving results to storage...')
    interaction_obj.store()

    print('Done. ')
