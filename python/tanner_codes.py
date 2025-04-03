import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt
import pym4ri

from experiments_settings import code_distance, code_dimension, smallest_SS_weight
from css_code_eval import MC_erasure_plog, MC_peeling_HGP, _HGP_peel, HGP, _fast_peel, stabilizer_search

from typing import Callable, Self
from tqdm import tqdm
import matplotlib.pyplot as plt

class TannerCode:
    def __init__(self, outer_code: sp.csr_array, local_subcodes: list[sp.csr_array]=None, 
                 local_subcode: sp.csr_array=None, order_assignments: list[np.ndarray]=None):
        """
        Initialize the TannerCode object.

        :param outer_graph: Base bipartite graph representing the relation between the bit nodes and the generalized check nodes. 
        :param local_subcodes: Specification of each local subcode separately. Length of the list must match number of check nodes. 
        :param local_subcode: Specification of the local subcode. If specified, all subcodes are the same. 
        :param order_assignments: Specifies the order of the bit nodes in the neighborhood of each check in its local subcode. 
                                  If None, the default order for each subcode is given by the order of the bit nodes in the outer graph. 
                                  The expected format is a permutation of [0, ..., r-1], where r is the length of the corresponding subcode. 
        """
        self.outer_code = outer_code
        num_tanner_checks = outer_code.shape[0]
        
        # If a single local subcode is specified, all Tanner checks carry the same local subcode. 
        if local_subcode is None:
            # use local_subcodes
            # Assert that you have specified as many subcodes as there are checks
            assert len(local_subcodes) == num_tanner_checks
            # And that each local code has the length compatible with the degree of its check node
            assert np.all(np.array([lsc.shape[1] for lsc in local_subcodes], dtype=np.int32) == np.diff(outer_code.indptr))
            self.local_subcodes = local_subcodes
            self.local_distances = np.array([code_distance(lsc) for lsc in self.local_subcodes])
        else:
            # override local_subcodes
            self.local_subcodes = [sp.csr_array(local_subcode) for _ in range(num_tanner_checks)]
            self.local_distances = np.array([code_distance(local_subcode)]*num_tanner_checks)

        # If no order assignment is specified, inherit implicit ordering from outer graph. 
        # Else, apply the order assignments to each subcode. 
        if order_assignments is not None:
            # Make sure you have as much assignments as there are check nodes. 
            assert len(order_assignments) == num_tanner_checks
            # And make sure each assignment is compatible with the degree of its check node. 
            assert np.all(np.array([len(ord) for ord in order_assignments], dtype=np.int32) == np.diff(outer_code.indptr))
            # (Cumbersome) make sure each order assignment is indeed a permutation
            assert all(len(np.unique(ord)) == len(ord) for ord in order_assignments)
            
            for lsc, ord in zip(self.local_subcodes, order_assignments):
                lsc.indices = ord[lsc.indices]
                for r in range(lsc.shape[0]):
                    lsc.indices[lsc.indptr[r]:lsc.indptr[r+1]].sort()

        # Compute PCM of the standard form code
        std_indptr, std_indices = outer_code.indptr, outer_code.indices
        std_data = np.concatenate([lsc.T.todense().reshape(*lsc.T.shape, 1) for lsc in self.local_subcodes])

        self.std_H = sp.bsr_array((std_data, std_indices, std_indptr)).asformat('csr')

    @classmethod
    def from_standard_code(cls, code: sp.csr_array, tanner_checks: list[np.ndarray]) -> Self:
        """
        Create a Tanner code from an existing code given in standard form, by grouping the checks into Tanner checks. 

        :param code: Normal code in standard form. 
        :param local_subcodes: Specification of which simple checks make up each Tanner check. 

        :return: TannerCode object. 
        """
        # Verify that tanner_checks defines a partition of the simple checks:
        row_order = np.concatenate(tanner_checks)
        unique_checks, inv_row_order, counts = np.unique(row_order, return_index=True, return_counts=True)
        assert np.all(unique_checks == np.arange(code.shape[0])) and np.all(counts == 1)
        
        code = code[row_order, :]

        outer_indptr, outer_indices = [0], []
        local_subcodes = []
        for tc in tanner_checks:
            local_support = np.unique(code[inv_row_order[tc], :].indices)
            local_subcodes.append(sp.csr_array(code[inv_row_order[tc], :][:, local_support].todense()))
            outer_indices.append(local_support)
            outer_indptr.append(len(local_support))

        outer_indptr = np.cumsum(outer_indptr)
        outer_indices = np.concatenate(outer_indices)
        outer_data = np.ones_like(outer_indices, dtype=np.int32)

        outer_code = sp.csr_array((outer_data, outer_indices, outer_indptr))

        return cls(outer_code, local_subcodes)


class TannerCodeHGP:
    def __init__(self, classic_tanner_C1: TannerCode, classic_tanner_C2: TannerCode=None):
        """
        Build the HGP of two classical Tanner codes, keeping track of the Tanner structure. 
        In order to run the Tanner peeling variants, we need structure of the standard form
        of the HGP code (std_Hx and std_Hz), for the standard peeling part, and for the Tanner
        peeling itself, we need to have the outer graph of the Hx and Hz codes upon aggregation
        of the check nodes. For AL's variant, we need only to inherit the local subcodes of the
        classical codes, whereas for my variants, we also need a mechanism to extract the rows
        of Hx/Hz relative to a Tanner check, including all the columns (both BB and CC blocks). 
        """
        if classic_tanner_C2 is None:
            classic_tanner_C2 = classic_tanner_C1

        # Recall that:
        # Hz = [I_n1 x H2 | H1.T x I_m2]
        # Hx = [H1 x I_n2 | I_m1 x H2.T]
        #      {BB qubits}|{ CC qubits }
        # to inherit the Tanner structure, we must partition the Hz and Hx matrices by keeping track of the 
        # Tanner structure of H2 and H1 respectively (column block relative to the BB qubits)
        
        # Define standard Hx and Hz structures
        # BB qubits
        # . .   - - Z ancillas
        # . .   - -
        # | |   * *
        # | |   * * CC qubits
        # X ancillas
        H1, H2 = classic_tanner_C1.std_H, classic_tanner_C2.std_H
        (m1, n1), (m2, n2), (t1, t2) = H1.shape, H2.shape, (classic_tanner_C1.outer_code.shape[0], classic_tanner_C2.outer_code.shape[0])
        I = lambda n: sp.eye_array(n, dtype=np.int32)
        self.std_Hz = sp.csr_array(sp.hstack([sp.kron(I(n1), H2), sp.kron(H1.T, I(m2))]).todense() & 1)
        self.std_Hx = sp.csr_array(sp.hstack([sp.kron(H1, I(n2)), sp.kron(I(m1), H2.T)]).todense() & 1)

        self.classic_C1_tanner_splitting = np.cumsum([0]+[lsc.shape[0] for lsc in classic_tanner_C1.local_subcodes], dtype=int)
        self.classic_C2_tanner_splitting = np.cumsum([0]+[lsc.shape[0] for lsc in classic_tanner_C2.local_subcodes], dtype=int)

        # Define outer graphs for Hz and Hx codes:
        self.outer_Hz = np.stack([
            np.bitwise_or.reduce(B, axis=1) 
            for B in np.split(self.std_Hz.todense().reshape(n1, H2.shape[0], -1), # shape (n1, m2, N)
                              self.classic_C2_tanner_splitting[1:-1], # section points over the 2nd axis
                              axis=1) # list of t2 elements of shape (n1, ?, N) (t2 = # of Tanner checks in C2)
                              ] # same list of t2 elements of shape (n1, N) (2nd axis collapsed by reduce operation)
                              ).swapaxes(0,1).reshape(n1*t2, -1) # shape (n1*t2, N)
        self.outer_Hz = sp.csr_array(self.outer_Hz & 1)
        
        self. outer_Hx = np.vstack([
            np.bitwise_or.reduce(B, axis=0) 
            for B in np.split(self.std_Hx.todense().reshape(H1.shape[0], n2, -1), # shape (m1, n2, N)
                              self.classic_C1_tanner_splitting[1:-1], # section points over the 1st axis
                              axis=0) # list of t1 elements of shape (?, n2, N) (t1 = # of Tanner checks in C1)
                              ] # same list of t1 elements of shape (n2, N) (1st axis collapsed by reduce operation)
                              ) # shape (t1*n2, N)
        self.outer_Hx = sp.csr_array(self.outer_Hx & 1)

        # Recover Tanner check structure from the classical graphs regarding only the BB qubits
        # Store local subcodes and local distances of the classical graphs        
        self.local_subcodes_C1 = classic_tanner_C1.local_subcodes
        self.local_distances_C1 = classic_tanner_C1.local_distances

        self.local_subcodes_C2 = classic_tanner_C2.local_subcodes
        self.local_distances_C2 = classic_tanner_C2.local_distances

        # Access the corresponding local subcodes for Hx/Hz by indexing those of C1/C2
        # Hx has t1 * n2 Tanner checks, so n2 copies of C1's Tanner checks, arranged col-wise (np.repeat)
        self.local_subcodes_BB_Hx = lambda i: self.local_subcodes_C1[i // n2] 
        self.local_distances_BB_Hx = np.repeat(self.local_distances_C1, n2)
        # Hz has n1 * t2 Tanner checks, so n1 copies of C2's Tanner checks, arranged row-wise (np.tile)
        self.local_subcodes_BB_Hz = lambda i: self.local_subcodes_C2[i % n1]
        self.local_distances_BB_Hz = np.tile(self.local_distances_C2, n1)

        self.shapes = ((m1, n1, t1), (m2, n2, t2))
        
        # This should be enough to implement Anthony's variant. 
        # std_H, outer_H, local_subcodes, local_distances in the classical case
        # std_Hz, std_Hx, outer_Hz, outer_Hx, local_subcodes_BBX/BBZ, local_distances_BBX/BBZ
        
        # Now, for my variants, I also need a way to recover the full local codes, i.e., that take into account
        # both the columns for the BB and CC qubits. Also, it might be necessary to precompute their distances :/
        # For now, let's worry about recovering these full local codes. 

        # For Hz:
        # i: index of the row in outer_Hz (tanner check idx) (n1*t2 rows)
        # r = i // n1, c = i % n1 (row and column of the tanner check in the outer Hz grid)
        # slice in std_Hz: 
        # start: r*m2 + self.classic_C2_tanner_splitting[c]
        # end (not included): start + C2.local_subcodes[c].shape[0]

        # For Hx: a bit more complicated... need a strided slice
        # i: index of the row in outer_Hx (tanner check idx) (t1*n2 rows)
        # r = i // n2, c = i % n2 (row and colunm of the tanner check in the outer Hx grid)
        # strided slice in std_Hx:
        # start: self.classic_C1_tanner_splitting[r]*n2 + c
        # end (not included): start + (C1.local_subcodes[r].shape[0])*n2
        # stride: n2

        # Now, we have the slices(i) of std_Hx/Hz corresponding to the i-th Tanner check, 
        # we have the support of the corresponding local subcode stored in outer_Hz/Hz[i]
        # That allows us to recover the local subcode matrix via std_Hx/Hz[slices(i), outer_Hx/Hz[i]]
        # From that we can also compute and store the local subcode distances. 
        # Design choice: precompute and store the distances, but don't store the local subcodes as separate objects.
        # When it comes to using them for decoding, extract them as needed. 
        
        # # self.local_subcodes_full_Hx = lambda i: self.std_Hx[???, <support>]
        # self.local_distances_full_Hx = np.zeros((t1*n2,))
        # # self.local_subcodes_full_Hz = lambda i: self.std_Hz[???, <support>]
        # self.local_distances_full_Hz = np.zeros((n1*t2,))

        # If we were to precompute distances: 
        # for i in range(t1*n2):
        #     self.local_distances_full_Hx[i] = code_distance(self.local_subcodes_full_Hx(i))
        # for i in range(n1*t2):
        #     self.local_distances_full_Hz[i] = code_distance(self.local_subcodes_full_Hz(i))
        

    def local_subcodes_full_Hx(self, i:int) -> tuple[sp.csr_array, np.ndarray]:
        """
        :param i: Index of the quantum Tanner check (row in std.outer_Hx)
        :return: The corresponding local subcode (all rows of the Tanner check, columns within the support)
        """
        (_, _, _), (_, n2, _) = self.shapes
        r, c = i // n2, i % n2
        start = self.classic_C1_tanner_splitting[r]*n2 + c
        end = start + self.local_subcodes_C1[r].shape[0]*n2
        stride = n2
        rows = self.std_Hx[start:end:stride, :]
        support = np.unique(rows.indices)
        return sp.csr_array(rows[:, support].todense()), support

    def local_subcodes_full_Hz(self, i:int) -> tuple[sp.csr_array, np.ndarray]:
        """
        :param i: Index of the quantum Tanner check (row in std.outer_Hz)
        :return: The corresponding local subcode (all rows of the Tanner check, columns within the support)
        """
        (_, _, _), (m2, _, t2) = self.shapes
        r, c = i // t2, i % t2
        start = r*m2 + self.classic_C2_tanner_splitting[c]
        end = start + self.local_subcodes_C2[c].shape[0]
        stride = 1
        rows = self.std_Hx[start:end:stride, :]
        support = np.unique(rows.indices)
        return sp.csr_array(rows[:, support].todense()), support

    def peel_0(self, erasure: np.ndarray, H: sp.csr_array, bootstrap_rows:tuple=None) -> bool:
        if bootstrap_rows is not None and np.any(erasure):
            start, stop, step = bootstrap_rows
            erased_cols = np.nonzero(erasure)[0]
            H_E = H[start:stop:step, erased_cols]
            dangling_checks = np.nonzero(np.diff(H_E.indptr) == 1)[0]
            if len(dangling_checks) > 0:
                erasure[erased_cols[H_E.indices[H_E.indptr[dangling_checks[0]]]]] = 0
            else:
                # This is the importance of the bootstrap
                return False
                
        return _fast_peel(erasure, H.indptr, H.indices)

    def peel_al(self, erasure: np.ndarray, only_X: bool=False) -> tuple[bool, bool]:
        """
        Anthony's proposed version of generalized peeling:
        Start by doing regular peeling until exhaustion. Next, apply the idea of
        peel_1 with respect to erasures on the BB patch only. If it changes anything, 
        do another round of regular peeling, alternate until both are exhausted. 
        """
        # Run the same thing for X-type error (Z-type stabilizers) and then for Z-type error, if not only_X. 
        (m1, n1, _), (m2, n2, _) = self.shapes
        N_BB = n1*n2

        # X-type error:
        erasure_X = erasure.copy()
        # Try to solve until both normal and generalized peeling are defeated (or succeed)
        
        normal_peeling_X_success = self.peel_0(erasure_X, self.std_Hz)
        
        while np.any(erasure_X):
            # Now start the generalized peeling
            erased_cols_BB = np.nonzero(erasure_X[:N_BB])[0]
            erased_cols_CC = np.nonzero(erasure_X[N_BB:])[0]
            outer_Hz_E_BB = self.outer_Hz[:, :N_BB][:, erased_cols_BB] # this is the tricky part for numba
            outer_Hz_E_CC = self.outer_Hz[:, N_BB:][:, erased_cols_CC] # this is the tricky part for numba
            # Filter out the Tanner checks that have CC erasures
            no_CC_erasure = np.nonzero(np.diff(outer_Hz_E_CC.indptr) == 0)[0]
 
            # Further filter out the Tanner checks whose BB erasure is longer than the local BB distance
            has_BB_erasure = np.nonzero(0 < np.diff(outer_Hz_E_BB[no_CC_erasure, :].indptr))[0]
            small_BB_erasure = np.nonzero(np.diff(outer_Hz_E_BB[no_CC_erasure[has_BB_erasure], :].indptr) 
                                                < self.local_distances_BB_Hz[no_CC_erasure[has_BB_erasure]])[0]
            
            dangling_Tanner_checks = no_CC_erasure[has_BB_erasure[small_BB_erasure]]
            
            if len(dangling_Tanner_checks) > 0:
                # Unerase local cluster of dangling bits
                # Basically solve the erasure in the neighborhood by ML with the local subcode
                # In practice, for benchmarking purposes, we can just consider all those bits to be 
                # unerased, since we know we can solve it by ML because the erasure is smaller than the distance. 
                dangling_Tanner_check = dangling_Tanner_checks[np.argmax(np.diff(outer_Hz_E_BB[dangling_Tanner_checks, :].indptr))]
                dangling_bits = erased_cols_BB[np.unique(outer_Hz_E_BB[[dangling_Tanner_check], :].indices)]
                erasure_X[dangling_bits] = 0

                # Afterwards, restart peeling, but focusing on the checks that are prone to have been unblocked
                # that is, checks whose support contains the unerased qubits. We can verify this now, and if none of them
                # have unblocked, then it's no use to go back to peeling. Else, start peeling until you block again
                r = dangling_Tanner_check // n1
                start, stop, step = r*m2, (r+1)*m2, 1
                self.peel_0(erasure_X, self.std_Hz, bootstrap_rows=(start, stop, step))

            else:
                # Report failure of the generalized peeling
                # If the generalized version fails even if just for X-type, overall both versions fail
                return False, False
            
        # Finished correcting X-type erasure (at this point the generalized version must have succeeded)
        if only_X:
            return normal_peeling_X_success, True
        else:
            # Z-type error:
            erasure_Z = erasure.copy()
            # Try to solve until both normal and generalized peeling are defeated (or succeed)
            normal_peeling_Z_success = self.peel_0(erasure_Z, self.std_Hx)
            
            while np.any(erasure_Z):
                # Now start the generalized peeling
                erased_cols_BB = np.nonzero(erasure_Z[:N_BB])[0]
                erased_cols_CC = np.nonzero(erasure_Z[N_BB:])[0]
                outer_Hx_E_BB = self.outer_Hx[:, :N_BB][:, erased_cols_BB] # this is the tricky part for numba
                outer_Hx_E_CC = self.outer_Hx[:, N_BB:][:, erased_cols_CC] # this is the tricky part for numba
                # Filter out the Tanner checks that have CC erasures
                no_CC_erasure = np.nonzero(np.diff(outer_Hx_E_CC.indptr) == 0)[0]
                # Further filter out the Tanner checks whose BB erasure is longer than the local BB distance
                has_BB_erasure = np.nonzero(0 < np.diff(outer_Hx_E_BB[no_CC_erasure, :].indptr))[0]
                small_BB_erasure = np.nonzero(np.diff(outer_Hx_E_BB[no_CC_erasure[has_BB_erasure], :].indptr) 
                                                    < self.local_distances_BB_Hx[no_CC_erasure[has_BB_erasure]])[0]
                dangling_Tanner_checks = no_CC_erasure[has_BB_erasure[small_BB_erasure]]
                
                if len(dangling_Tanner_checks) > 0:
                    # Unerase local cluster of dangling bits
                    # Basically solve the erasure in the neighborhood by ML with the local subcode
                    # In practice, for benchmarking purposes, we can just consider all those bits to be 
                    # unerased, since we know we can solve it by ML because the erasure is smaller than the distance. 
                    dangling_Tanner_check = dangling_Tanner_checks[np.argmax(np.diff(outer_Hx_E_BB[dangling_Tanner_checks, :].indptr))]
                    dangling_bits = erased_cols_BB[np.unique(outer_Hx_E_BB[[dangling_Tanner_check], :].indices)]
                    erasure_Z[dangling_bits] = 0

                    # Afterwards, restart peeling, but focusing on the checks that are prone to have been unblocked
                    # that is, checks whose support contains the unerased qubits. We can verify this now, and if none of them
                    # have unblocked, then it's no use to go back to peeling. Else, start peeling until you block again
                    c = dangling_Tanner_check % n2
                    start, stop, step = c, c+m1*n2, n2
                    self.peel_0(erasure_Z, self.std_Hx, bootstrap_rows=(start, stop, step))

                else:
                    # Report failure of the generalized peeling
                    # If the generalized version fails even if just for Z-type, overall both versions fail
                    return False, False
                        
        # Finished correcting Z-type erasure (at this point the generalized version must have succeeded)
        normal_peeling_success = normal_peeling_X_success and normal_peeling_Z_success
        return normal_peeling_success, True
    
    def peel_v3(self, erasure: np.ndarray, only_X: bool=False) -> tuple[bool, bool]:
        """
        Anthony's (and mine) proposed version of generalized peeling:
        Start by doing regular peeling until exhaustion. Next, apply the idea of
        peel_3: a Tanner check is dangling if you can determine any nonzero number of
        erased bits by ML decoding. In practice, you will always attempt to do so (costly). 
        """
        # Run the same thing for X-type error (Z-type stabilizers) and then for Z-type error, if not only_X. 
        # X-type error:
        erasure_X = erasure.copy()
        
        # Try to solve until both normal and generalized peeling as well as pruning fail (or succeed)
        normal_peeling_X_success, gen_peeling_X_success = self._peel_v3(erasure_X, self.std_Hz, self.outer_Hz, 
                                                                        self.local_subcodes_full_Hz)
        if not gen_peeling_X_success:
            return False, False
        # Finished correcting X-type erasure (at this point the generalized version must have succeeded)
        if only_X:
            return normal_peeling_X_success, True
        else:
            # Z-type error:
            erasure_Z = erasure.copy()
            
            # Try to solve until both normal and generalized peeling are defeated (or succeed)
            normal_peeling_Z_success, gen_peeling_Z_success = self._peel_v3(erasure_Z, self.std_Hx, self.outer_Hx, 
                                                                            self.local_subcodes_full_Hx)
            if not gen_peeling_Z_success:
                return False, False
            # Finished correcting Z-type erasure (at this point the generalized version must have succeeded)
            normal_peeling_success = normal_peeling_X_success and normal_peeling_Z_success
            return normal_peeling_success, True

    def _peel_v3(self, erasure: np.ndarray, std_H: sp.csr_array, outer_H: sp.csr_array, local_subcodes_full:Callable) -> tuple[bool, bool]:     
        # Try to solve until both normal and generalized peeling are defeated (or succeed)
        normal_peeling_success = self.peel_0(erasure, std_H)
        
        # Now start the generalized peeling
        while np.any(erasure):
            # Search for a "dangling" Tanner check by testing each one of them. 
            # Select erased columns
            outer_H_E = outer_H[:, erasure] # this is the tricky part for numba
            # Filter out the Tanner checks that have no erasures
            has_erasure = np.nonzero(0 < np.diff(outer_H_E.indptr))[0]
            
            # Now we have to iterate over all Tanner checks that have any erasure 
            # until we find one that can at least partially solve its erasures
            unblocked = False
            for tanner_check in has_erasure:
                # Fetch local code and its support
                local_H, supp = local_subcodes_full(tanner_check)
                # Further select erased columns
                local_H_E = local_H[:, erasure[supp]]
                # Convert the pcm to a gen matrix: find a basis of its kernel
                ML_solution = pym4ri.chk2gen(local_H_E.astype(bool).todense())
                # The determined bits are the rows of this gen matrix for which all entries are zero
                determined_bits = np.bitwise_not(np.bitwise_or.reduce(ML_solution, axis=1))
                # Find the corresponding global index of the determined bits
                erased_within_support = np.nonzero(erasure[supp])[0]
                determined_bits_global = supp[erased_within_support][determined_bits]
                # Unerase the determined bits:
                erasure[determined_bits_global] = False

                # If any bits were unerased, then we have unblocked, and we go back to normal peeling
                if np.any(determined_bits):
                    unblocked = True
                    break
                
            # If no bits were unerased by any of the Tanner checks, we've failed. 
            if unblocked:
                self.peel_0(erasure, std_H)
            else:
                return False, False
        
        # At this point, the generalized version must have succeeded.
        return normal_peeling_success, True

    def peel_v4(self, erasure: np.ndarray, pruning_depth: int=2, only_X: bool=False) -> tuple[bool, bool, list[bool]]:
        """
        Peeling v4 is meant to be peeling v3 + pruning. 
        The pruning depth controls how many generators you consider to multiply to find a stabilizer that fits in the erasure. 
        """
        # Run the same thing for X-type error (Z-type stabilizers) and then for Z-type error, if not only_X. 
        # X-type error:
        erasure_X = erasure.copy()

        # Try to solve until both normal and generalized peeling as well as pruning fail (or succeed)
        normal_peeling_X_success, gen_peeling_X_success, pruning_X_success = \
            self._peel_v4(erasure_X, self.std_Hz, self.outer_Hz, self.local_subcodes_full_Hz, self.std_Hx, pruning_depth)
            
        if not pruning_X_success[-1]:
            return False, False, [False]*pruning_depth
        # At this point, pruning over X must have succeeded
        if only_X:
            return normal_peeling_X_success, gen_peeling_X_success, pruning_X_success
        else:
            # Z-type error: 
            erasure_Z = erasure.copy()

            # Try to solve until both normal and generalized peeling as well as pruning fail (or succeed)
            normal_peeling_Z_success, gen_peeling_Z_success, pruning_Z_success = \
                self._peel_v4(erasure_Z, self.std_Hx, self.outer_Hx, self.local_subcodes_full_Hx, self.std_Hz, pruning_depth)
            
            if not pruning_Z_success[-1]:
                return False, False, [False]*pruning_depth
            # At this point, pruning over Z must have succeeded
            normal_peeling_success = normal_peeling_X_success and normal_peeling_Z_success
            gen_peeling_success = gen_peeling_X_success and gen_peeling_Z_success
            pruning_success = [pXs and pZs for pXs, pZs in zip(pruning_X_success, pruning_Z_success)]
            return normal_peeling_success, gen_peeling_success, pruning_success

    def _peel_v4(self, erasure: np.ndarray, std_H: sp.csr_array, outer_H: sp.csr_array, 
                 local_subcodes_full:Callable, std_H_conj: sp.csr_array, pruning_depth: int) -> tuple[bool, bool, list[bool]]:  
        
        normal_peeling_success, gen_peeling_success = self._peel_v3(erasure, std_H, outer_H, local_subcodes_full)

        # After normal and generalized peeling fail (or succeed), start pruning
        found_at_depth = [True]*pruning_depth
        while np.any(erasure):
            # If generalized peeling didn't work, look for a nontrivial stabilizer inside the erasure
            found_at_depth, stabilizer = stabilizer_search(std_H_conj.todense(), erasure, pruning_depth)
            if found_at_depth[-1]:
                # If found, arbiter the correction over a qubit inside its support (and unerase it)
                gauge_qubit = np.nonzero(stabilizer)[0][-1]
                erasure[gauge_qubit] = 0
                # Go back to peeling
                self._peel_v3(erasure, std_H, outer_H, local_subcodes_full)
            else:
                # If no stabilizer was found, pruning has failed. 
                break
        
        return normal_peeling_success, gen_peeling_success, found_at_depth
    
    def peel_v5(self, erasure: np.ndarray, pruning_depth: int=1, only_X: bool=False) -> tuple[bool, list[bool], bool]:
        """
        Peeling v5 is basically v4 but in a different resolution order: peeling -> pruning -> gen. peeling
        """
        # X-type error:
        erasure_X = erasure.copy()

        # Try to solve until both peeling, pruning and generalized peeling fail (or succeed)
        normal_peeling_X_success, pruning_X_success, gen_peeling_X_success = \
            self._peel_v5(erasure_X, self.std_Hz, self.outer_Hz, self.local_subcodes_full_Hz, self.std_Hx, pruning_depth)
            
        if not gen_peeling_X_success:
            return False, [False]*pruning_depth, False
        # At this point, decoding the X erasure must have succeeded
        if only_X:
            return normal_peeling_X_success, pruning_X_success, gen_peeling_X_success
        else:
            # Z-type error: 
            erasure_Z = erasure.copy()

            # Try to solve until both peeling, pruning and generalized peeling fail (or succeed)
            normal_peeling_Z_success, pruning_Z_success, gen_peeling_Z_success = \
                self._peel_v5(erasure_Z, self.std_Hx, self.outer_Hx, self.local_subcodes_full_Hx, self.std_Hz, pruning_depth)
            
            if not gen_peeling_Z_success:
                return False, [False]*pruning_depth, False
            
            # At this point, decoding the Z erasure must have succeeded
            normal_peeling_success = normal_peeling_X_success and normal_peeling_Z_success
            pruning_success = [pXs and pZs for pXs, pZs in zip(pruning_X_success, pruning_Z_success)]
            gen_peeling_success = gen_peeling_X_success and gen_peeling_Z_success
            return normal_peeling_success, pruning_success, gen_peeling_success

    def _peel_v5(self, erasure: np.ndarray, std_H: sp.csr_array, outer_H: sp.csr_array, 
                 local_subcodes_full:Callable, std_H_conj: sp.csr_array, pruning_depth: int) -> tuple[bool, list[bool], bool]:  
        # Do normal peeling first
        peeling_success = self.peel_0(erasure, std_H)
        if not peeling_success: # NOTE: keep track of smallest failure set
            self.min_SS_size['peel'] = min(self.min_SS_size['peel'], np.count_nonzero(erasure))

        # After peeling stops, do pruning
        pruning_success = [True]*pruning_depth
        while np.any(erasure):
            # If peeling didn't work, look for a nontrivial stabilizer inside the erasure
            # If pruning failed at max depth, never try it again. 
            if pruning_success[-1]:
                found_at_depth, stabilizer = stabilizer_search(std_H_conj.todense(), erasure, pruning_depth)
                pruning_success = [ps and fd for ps, fd in zip(pruning_success, found_at_depth)]
                for i, ps in enumerate(pruning_success): # NOTE: keep track of smallest failure set
                    if not ps:
                        self.min_SS_size['prun'][i] = min(self.min_SS_size['prun'][i], np.count_nonzero(erasure))
                # If found, arbiter the correction over a qubit inside its support (and unerase it)
                gauge_qubits = np.nonzero(stabilizer)[0]
                gauge_qubit = min(gauge_qubits, key=lambda qb: std_H[:, erasure][std_H[:, [qb]].astype(bool).todense().flatten(), :].sum(axis=1).min())
                # gauge_qubit = gauge_qubits[-1]
                erasure[gauge_qubit] = 0
                # Go back to peeling
                self.peel_0(erasure, std_H)
            else:
                # If no stabilizer was found, pruning has failed. 
                # Search for a "dangling" Tanner check by testing each one of them. 
                # Select erased columns
                outer_H_E = outer_H[:, erasure] # this is the tricky part for numba
                # Filter out the Tanner checks that have no erasures
                has_erasure = np.nonzero(0 < np.diff(outer_H_E.indptr))[0]
                
                # Now we have to iterate over all Tanner checks that have any erasure 
                # until we find one that can at least partially solve its erasures
                unblocked = False
                for tanner_check in has_erasure:
                    # Fetch local code and its support
                    local_H, supp = local_subcodes_full(tanner_check)
                    # Further select erased columns
                    local_H_E = local_H[:, erasure[supp]]
                    # Convert the pcm to a gen matrix: find a basis of its kernel
                    ML_solution = pym4ri.chk2gen(local_H_E.astype(bool).todense())
                    # The determined bits are the rows of this gen matrix for which all entries are zero
                    determined_bits = np.bitwise_not(np.bitwise_or.reduce(ML_solution, axis=1))
                    # Find the corresponding global index of the determined bits
                    erased_within_support = np.nonzero(erasure[supp])[0]
                    determined_bits_global = supp[erased_within_support][determined_bits]
                    # Unerase the determined bits:
                    erasure[determined_bits_global] = False

                    # If any bits were unerased, then we have unblocked, and we go back to normal peeling
                    if np.any(determined_bits):
                        unblocked = True
                        break
                    
                # If no bits were unerased by any of the Tanner checks, we've failed. 
                if unblocked:
                    self.peel_0(erasure, std_H)
                else:
                    self.min_SS_size['gen peel'] = min(self.min_SS_size['gen peel'], np.count_nonzero(erasure)) # NOTE: keep track of smallest failure set
                    return peeling_success, pruning_success, False

        return peeling_success, pruning_success, True

    def gen_peel_benchmark(self, p_vals:np.ndarray[float], max_num_trials:int, min_num_fails:int=None, pruning_depth: int=1,
                          only_X:bool=False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[int, dict[str, np.ndarray]]]:
        (m1, n1, _), (m2, n2, _) = self.shapes
        N = n1*n2 + m1*m2
        
        mean = lambda n_fail, n_total: n_fail/n_total
        std = lambda n_fail, n_total: ((n_fail*(n_total - n_fail)) / (n_total*(n_total - 1)))**.5
        eb = lambda n_fail, n_total: 1.96*std(n_fail, n_total)/np.sqrt(n_total)
        
        normal_peeling_stats = {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)}
        normal_peeling_failures = np.zeros_like(p_vals)

        pruning_stats = {d+1: {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)} for d in range(pruning_depth)}
        pruning_failures = {d+1: np.zeros_like(p_vals) for d in range(pruning_depth)}
        
        generalized_peeling_stats = {"ler": np.zeros_like(p_vals), "ler_eb": np.zeros_like(p_vals)}
        generalized_peeling_failures = np.zeros_like(p_vals)

        self.min_SS_size = {"peel": N, "prun": [N]*pruning_depth, "gen peel": N}
        
        if min_num_fails is None:
                min_num_fails = max_num_trials
        
        for i, er in enumerate(p_vals):
            num_trials = max_num_trials
            for t in (pbar := tqdm(range(max_num_trials))):
                descr = f'#failures: Peel. {normal_peeling_failures[i]:.0f}'
                for d, pruning_failures_d in pruning_failures.items():                    
                    descr += f'| Peel.+Prun. D={d} = {pruning_failures_d[i]:.0f}'
                descr += f'| Gen. Peel. = {generalized_peeling_failures[i]:.0f}'
                pbar.set_description(descr)

                erasure = npr.rand(N) < er
                normal_peeling_success, pruning_success, generalized_peeling_success = self.peel_v5(erasure, pruning_depth=pruning_depth, only_X=only_X)

                normal_peeling_failures[i] += 1 if not normal_peeling_success else 0
                for pruning_failures_d, pruning_success_d in zip(pruning_failures.values(), pruning_success):
                    pruning_failures_d[i] += 1 if not pruning_success_d else 0
                generalized_peeling_failures[i] += 1 if not generalized_peeling_success else 0
                
                if generalized_peeling_failures[i] >= min_num_fails:
                    num_trials = t+1
                    break

            normal_peeling_stats["ler"][i] = mean(normal_peeling_failures[i], num_trials)
            normal_peeling_stats["ler_eb"][i] = eb(normal_peeling_failures[i], num_trials)

            for pruning_stats_d, pruning_failures_d in zip(pruning_stats.values(), pruning_failures.values()):
                pruning_stats_d["ler"][i] = mean(pruning_failures_d[i], num_trials)
                pruning_stats_d["ler_eb"][i] = eb(pruning_failures_d[i], num_trials)

            generalized_peeling_stats["ler"][i] = mean(generalized_peeling_failures[i], num_trials)
            generalized_peeling_stats["ler_eb"][i] = eb(generalized_peeling_failures[i], num_trials)

        return normal_peeling_stats, pruning_stats, generalized_peeling_stats



if __name__ == '__main__':
    # Modify this to make it a precomputation script. 
    # Save all the TannerCodeHGP objects with all precomputed structure 
    # as pickle files to be imported by the benchmarking script

    cyclic_hamming = sp.csr_array([[1,0,1,0,1,0,1], 
                                   [0,1,1,0,1,1,0], 
                                   [0,0,0,1,1,1,1]])
    
    # print(f'[n={cyclic_hamming.shape[1]}, k={code_dimension(cyclic_hamming)}, d={code_distance(cyclic_hamming)}]')
    
    K8 = nx.complete_graph(8)
    spkdwhl_H = nx.incidence_matrix(K8, dtype=np.int32).asformat('csr')    

    # [28, 6, 10] code with [24, 2, 13] transpose
    # tanner_code = TannerCode(outer_graph=spkdwhl_H, 
    #                          local_subcode=cyclic_hamming, 
    #                          order_assignments=[np.roll(np.concatenate([[0], np.roll(np.arange(1,7), shift=2*k)]), shift=3) for k in range(8)])

    # [28, 6, 10] code with [24, 2, 8] transpose
    # tanner_code = TannerCode(outer_graph=spkdwhl_H, 
    #                          local_subcode=cyclic_hamming, 
    #                          order_assignments=[np.roll(np.concatenate([[0], np.roll(np.arange(1,7), shift=3*k)]), shift=3) for k in range(8)])

    # [28, 5, 10] code with [24, 1, 13] tranpose
    # tanner_code = TannerCode(outer_graph=spkdwhl_H, 
    #                          local_subcode=cyclic_hamming, 
    #                          order_assignments=[np.roll(np.concatenate([[0], np.roll(np.arange(1,7), shift=4*k)]), shift=3) for k in range(8)])

    # Template recipe:
    # tanner_code = TannerCode(outer_code=spkdwhl_H, 
    #                          local_subcode=cyclic_hamming, 
    #                          order_assignments=[np.roll(np.concatenate([[0], np.roll(np.arange(1,7), shift=3 if k>0 else 0)]), shift=k) for k in range(8)])


    
    ord_ass_28_4_13 = [
        np.roll(np.concatenate([np.array([0]), (1+np.roll(np.arange(7), shift=1))]), shift=k) for k in range(8)
    ]

    ord_ass_28_6_10 = [
        [0, 1, 2, 4, 3, 6, 7, 5], 
        [4, 0, 5, 6, 7, 3, 2, 1], 
        [4, 1, 0, 5, 6, 7, 3, 2], 
        [4, 2, 1, 0, 5, 6, 7, 3], 
        [4, 3, 2, 1, 0, 5, 6, 7], 
        [4, 7, 3, 2, 1, 0, 5, 6], 
        [4, 6, 7, 3, 2, 1, 0, 5], 
        [4, 5, 6, 7, 3, 2, 1, 0], 
    ]

    ord_ass_28_10_6 = [
        [u^v for v in range(8)] for u in range(8)
    ]

    order_assignment_list = [ord_ass_28_4_13, ord_ass_28_6_10, ord_ass_28_10_6]


    def tanner_code_K8_Hamming(order_assignment: list[list[int]]) -> sp.csr_array:
        K8 = nx.complete_graph(8)
            
        indptr, indices, data = [0], [], []
        for u in K8.nodes:
            count = 0
            for i, e in enumerate(K8.edges):
                if u in e:
                    count += 1
                    indices.append(i)
                    v = [w for w in e if w != u][0]
                    x = np.unpackbits(np.uint8(order_assignment[u][v]), count=3, bitorder='little').astype(np.int32).reshape(3, 1)
                    data.append(x)

            indptr.append(count)
        indptr = np.cumsum(indptr)
        indices = np.array(indices)
        data = np.stack(data, axis=0)

        return sp.csr_array(sp.bsr_array((data, indices, indptr)).todense())

    code = tanner_code_K8_Hamming(order_assignment_list[0])
    new_tanner_code = TannerCode.from_standard_code(code, [3*k + np.arange(3) for k in range(8)])
    # Compute parameters:
    # print(f"""[n={new_tanner_code.std_H.shape[1]}, k={code_dimension(new_tanner_code.std_H)}, 
    #       d={code_distance(new_tanner_code.std_H)}, s={smallest_SS_weight(new_tanner_code.std_H)[0]}]""")
    # print(f"""[n={new_tanner_code.std_H.T.shape[1]}, k={code_dimension(new_tanner_code.std_H.T)}, 
    #       d={code_distance(new_tanner_code.std_H.T)}, s={smallest_SS_weight(new_tanner_code.std_H.T)[0]}]]""")

    new_tanner_hgp = TannerCodeHGP(new_tanner_code)
    # print(f"""Hx is ({new_tanner_hgp.std_Hx.sum(axis=1).min()}-{new_tanner_hgp.std_Hx.sum(axis=1).max()}, 
    #                {new_tanner_hgp.std_Hx.sum(axis=0).min()}-{new_tanner_hgp.std_Hx.sum(axis=0).max()})-LDPC""")
    # print(f"""Hz is ({new_tanner_hgp.std_Hz.sum(axis=1).min()}-{new_tanner_hgp.std_Hz.sum(axis=1).max()}, 
    #                {new_tanner_hgp.std_Hz.sum(axis=0).min()}-{new_tanner_hgp.std_Hz.sum(axis=0).max()})-LDPC""")
    
    theta = bpt.from_biadjacency_matrix(new_tanner_code.std_H, create_using=nx.MultiGraph)
    c = [n for n, b in theta.nodes(data='bipartite') if b == 0]
    v = [n for n, b in theta.nodes(data='bipartite') if b == 1]
    H = sp.csr_array(bpt.biadjacency_matrix(theta, row_order=sorted(c), column_order=sorted(v)).todense() & 1)
    Hx, Hz = HGP(H)
    assert np.all(Hx.todense() == new_tanner_hgp.std_Hx.todense())
    assert np.all(Hz.todense() == new_tanner_hgp.std_Hz.todense())
    
    erasure_rate = np.array([0.15, 0.25, 0.35])
    npr.seed(200)
    normal_peeling_stats, pruning_stats, generalized_peeling_stats = new_tanner_hgp.gen_peel_benchmark(erasure_rate, 
                                                                                                       max_num_trials=int(1e3), 
                                                                                                       pruning_depth=1)
    
    theta = bpt.from_biadjacency_matrix(new_tanner_code.std_H, create_using=nx.MultiGraph)
    # ML_results = MC_erasure_plog(int(1e4), state=theta, p_vals=erasure_rate)
    
    # Pure peeling: 
    npr.seed(200)
    peeling_results = MC_peeling_HGP(int(1e3), state=theta, p_vals=erasure_rate)

    print(new_tanner_hgp.min_SS_size)
    
    fig, ax = plt.subplots(1, 1)
    ax.set_yscale('log')
    # ax.errorbar(erasure_rate, ML_results["mean"], 1.96*ML_results["std"]/1e2, label="ML")
    ax.errorbar(erasure_rate, normal_peeling_stats["ler"], normal_peeling_stats["ler_eb"], label="normal")
    for d, stats in pruning_stats.items():
        ax.errorbar(erasure_rate, stats["ler"], stats["ler_eb"], label=f"pruning depth={d}")
    ax.errorbar(erasure_rate, generalized_peeling_stats["ler"], generalized_peeling_stats["ler_eb"], label="generalized")
    ax.errorbar(erasure_rate, peeling_results["mean"], 1.96*peeling_results["std"]/np.sqrt(1e3), label="peeling that works", linestyle='--')
    plt.legend()
    plt.show()
    

