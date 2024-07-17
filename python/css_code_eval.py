import numpy as np
import numpy.random as npr
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp

import pym4ri as m4ri

def CSS_HGP_code_from_state(state: nx.MultiGraph) -> tuple[sp.sparray, sp.sparray, int]:
    # Convert multigraph state to simple graph
    G = nx.Graph(state)
    
    # Extract biadjacency matrix from the Tanner graph of the classical code
    c, v = bpt.sets(G)
    H = bpt.biadjacency_matrix(G, row_order=sorted(c), column_order=sorted(v), dtype=np.bool_)
    m, n = H.shape
    
    # Compute HGP code from classical code
    Hx = sp.hstack([sp.kron(sp.eye_array(m), H.T), # Im x H2
                    sp.kron(H, sp.eye_array(n)),   # H1 x In  
                   ], dtype=np.bool_).tocsc() # [ Im x H2 | H1 x In ] } m*n rows

    Hz = sp.hstack([sp.kron(H.T, sp.eye_array(m)), # H1'x Im
                    sp.kron(sp.eye_array(n), H),   # In x H2'
                   ], dtype=np.bool_).tocsc() # [ H1'x Im | In x H2'] } m*n rows

    # Return CSS matrices and number of qubits
    return Hx, Hz, m*m + n*n

def MC_erasure_plog_fixed_p_only_X(num_trials: int, state: nx.MultiGraph, p: float) -> float:
    return MC_erasure_plog_only_X(num_trials, state, [p])['mean'][0]

def MC_erasure_plog_lower_bound_fixed_p_only_X(num_trials: int, state: nx.MultiGraph, p: float) -> float:
    return MC_erasure_plog_lower_bound_only_X(num_trials, state, [p])['mean'][0]

def MC_erasure_plog_fixed_p(num_trials: int, state: nx.MultiGraph, p: float) -> float:
    return MC_erasure_plog(num_trials, state, [p])['mean'][0]

def MC_erasure_plog_lower_bound_fixed_p(num_trials: int, state: nx.MultiGraph, p: float) -> float:
    return MC_erasure_plog_lower_bound(num_trials, state, [p])['mean'][0]

def MC_erasure_plog_lower_bound_symplectic_fixed_p(num_trials: int, state: nx.MultiGraph, p: float) -> float:
    return MC_erasure_plog_lower_bound_symplectic(num_trials, state, [p])['mean'][0]

def MC_erasure_plog(num_trials: int, state: nx.MultiGraph, p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Compute the parity-check matrices of Cx^\perp and Cz^\perp
    etaHz = m4ri.gen2chk(Hz.todense())
    etaHx = m4ri.gen2chk(Hx.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < p

            # Generate the Cx and Cz intersections with the erasure support
            gammaHxE = m4ri.chk2gen(Hx[:, E].todense())
            gammaHzE = m4ri.chk2gen(Hz[:, E].todense())
            
            # Check for the existence of logical errors hiding in the erasure
            z_logical_error = m4ri.gf2_mul(etaHz[:, E], gammaHxE).any()
            x_logical_error = m4ri.gf2_mul(etaHx[:, E], gammaHzE).any()
            
            # Failure criterion: the erasure hides a logical error of either type
            failure = z_logical_error or x_logical_error
            trials.append(failure)

        trials = np.array(trials)
        est_plog['mean'].append(trials.mean())
        est_plog['std'].append(trials.std())
        
    return est_plog

def MC_erasure_plog_importance_sampling(num_trials: int, state: nx.MultiGraph, p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Compute the parity-check matrices of Cx^\perp and Cz^\perp
    etaHz = m4ri.gen2chk(Hz.todense())
    etaHx = m4ri.gen2chk(Hx.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < .5

            # Generate the Cx and Cz intersections with the erasure support
            gammaHxE = m4ri.chk2gen(Hx[:, E].todense())
            gammaHzE = m4ri.chk2gen(Hz[:, E].todense())
            
            # Check for the existence of logical errors hiding in the erasure
            z_logical_error = m4ri.gf2_mul(etaHz[:, E], gammaHxE).any()
            x_logical_error = m4ri.gf2_mul(etaHx[:, E], gammaHzE).any()
            
            # Failure criterion: the erasure hides a logical error of either type
            failure = z_logical_error or x_logical_error
            trials.append(failure * (p/(1-p))**E.sum() * (2-2*p)**N_qubits)

        trials = np.array(trials)
        est_plog['mean'].append(trials.mean())
        est_plog['std'].append(trials.std())
        
    return est_plog

def MC_erasure_plog_lower_bound(num_trials: int, state: nx.MultiGraph, 
                               p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Rank of the stabilizer matrix H = [Hx \\ Hz] in symplectic form
    rank_H = m4ri.rank(Hx.todense()) + m4ri.rank(Hz.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < p
            E_ = np.bitwise_not(E)

            # Compute rank(H_E) and rank(H_E') in symplectic form
            rank_H_E = m4ri.rank(Hx[:, E].todense()) + m4ri.rank(Hz[:, E].todense())
            rank_H_E_ = m4ri.rank(Hx[:, E_].todense()) + m4ri.rank(Hz[:, E_].todense())
            
            # Compute dimension of the normalizer and stabilizer groups
            dim_NSE = 2*E.sum() - rank_H_E
            dim_SE = rank_H - rank_H_E_

            # Failure criterion: necessary condition for correctability doesn't hold
            failure = dim_NSE > dim_SE
            trials.append(failure)

        trials = np.array(trials)
        est_plog['mean'].append(trials.mean())
        est_plog['std'].append(trials.std())
        
    return est_plog

def MC_erasure_plog_lower_bound_symplectic(num_trials: int, state: nx.MultiGraph, 
                               p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Rank of the stabilizer matrix H = [Hx \\ Hz] in symplectic form
    H = sp.block_array([[Hx, None], 
                        [None, Hz]]).tocsc()
    rank_H = m4ri.rank(H.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < p
            E = np.hstack([E, E])
            E_ = np.bitwise_not(E)

            # Compute rank(H_E) and rank(H_E') in symplectic form
            rank_H_E = m4ri.rank(H[:, E].todense())
            rank_H_E_ = m4ri.rank(H[:, E_].todense())
            
            # Compute dimension of the normalizer and stabilizer groups
            dim_NSE = E.sum() - rank_H_E
            dim_SE = rank_H - rank_H_E_

            # Failure criterion: necessary condition for correctability doesn't hold
            failure = dim_NSE > dim_SE
            trials.append(failure)

        trials = np.array(trials)
        est_plog['mean'].append(trials.mean())
        est_plog['std'].append(trials.std())
        
    return est_plog

def MC_erasure_plog_only_X(num_trials: int, state: nx.MultiGraph, 
                               p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Compute the parity-check matrices of Cx^\perp
    etaHx = m4ri.gen2chk(Hx.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < p

            # Generate the Cz intersection with the erasure support
            gammaHzE = m4ri.chk2gen(Hz[:, E].todense())
            
            # Check for the existence of x logical errors hiding in the erasure
            x_logical_error = m4ri.gf2_mul(etaHx[:, E], gammaHzE).any()
            
            # Verify only logical errors of type x
            trials.append(x_logical_error)

        trials = np.array(trials)
        
        # assume P[failure] = P[x_log_err or z_log_err] = 2*P[x_log_err] - P[x_log_err]^2
        m, s = trials.mean(), trials.std()
        
        est_plog['mean'].append(m)
        # est_plog['mean'].append(2*m - m*m)
        est_plog['std'].append(s*np.sqrt(2 - m - s*s))
        
    return est_plog


def MC_erasure_plog_lower_bound_only_X(num_trials: int, state: nx.MultiGraph, 
                               p_vals: list[float]=None) -> dict:
    # Compute CSS code for the HGP induced by the state
    Hx, Hz, N_qubits = CSS_HGP_code_from_state(state)
    
    # Rank of the stabilizer matrix Hx
    rank_H = m4ri.rank(Hx.todense())

    # Register mean and std of the Monte Carlo estimates
    est_plog = {'mean': [], 'std': []}

    for p in p_vals:
        trials = []
        for _ in range(num_trials):
            # IID erasure with probability p (physical error rate)
            E = npr.rand(N_qubits) < p
            E_ = np.bitwise_not(E)

            # Compute rank(H_E) and rank(H_E') in symplectic form
            rank_H_E = m4ri.rank(Hz[:, E].todense())
            rank_H_E_ = m4ri.rank(Hx[:, E_].todense())
            
            # Compute dimension of the normalizer and stabilizer groups
            dim_NSE = E.sum() - rank_H_E
            dim_SE = rank_H - rank_H_E_

            # Verify rank condition only for x errors
            trials.append(dim_NSE > dim_SE)

        trials = np.array(trials)
        
        # assume P[failure] = P[x_log_err or z_log_err] = 2*P[x_log_err] - P[x_log_err]^2
        m, s = trials.mean(), trials.std()
        
        est_plog['mean'].append(m)
        # est_plog['mean'].append(2*m - m*m)
        est_plog['std'].append(s*np.sqrt(2 - m - s*s))
        
    return est_plog