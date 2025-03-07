import numpy as np
import numpy.random as npr
import scipy.sparse as sp
import networkx as nx
import networkx.algorithms.bipartite as bpt
import pym4ri

from experiments_settings import code_distance, code_dimension

class TannerCode:
    def __init__(self, outer_graph: sp.csr_array, local_subcodes: list[sp.csr_array]=None, 
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
        self.codelength = outer_graph.shape[1]
        self.num_tanner_checks = outer_graph.shape[0]
        
        # If a single local subcode is specified, all Tanner checks carry the same local subcode. 
        if local_subcode is None:
            # use local_subcodes
            # Assert that you have specified as many subcodes as there are checks
            assert len(local_subcodes) == self.num_tanner_checks
            # And that each local code has the length compatible with the degree of its check node
            assert np.all(np.array([lsc.shape[1] for lsc in local_subcodes], dtype=np.int32) == np.diff(outer_graph.indptr))
            self.local_subcodes = local_subcodes
        else:
            # override local_subcodes
            self.local_subcodes = [sp.csr_array(local_subcode) for _ in range(self.num_tanner_checks)]

        # If no order assignment is specified, inherit implicit ordering from outer graph. 
        # Else, apply the order assignments to each subcode. 
        if order_assignments is not None:
            # Make sure you have as much assignments as there are check nodes. 
            assert len(order_assignments) == self.num_tanner_checks
            # And make sure each assignment is compatible with the degree of its check node. 
            assert np.all(np.array([len(ord) for ord in order_assignments], dtype=np.int32) == np.diff(outer_graph.indptr))
            # (Cumbersome) make sure each order assignment is indeed a permutation
            assert all(len(np.unique(ord)) == len(ord) for ord in order_assignments)
            
            for lsc, ord in zip(self.local_subcodes, order_assignments):
                lsc.indices = ord[lsc.indices]
                for r in range(lsc.shape[0]):
                    lsc.indices[lsc.indptr[r]:lsc.indptr[r+1]].sort()

        # Compute PCM of the standard form code
        std_indptr, std_indices, std_data = [np.array([0], dtype=np.int32)], [], []

        for idx, lsc in enumerate(self.local_subcodes):
            # Concatenate the weight of each row for each local subcode
            std_indptr.append(np.diff(lsc.indptr))

            # Find the support of the local code
            support = outer_graph.indices[outer_graph.indptr[idx]:outer_graph.indptr[idx+1]]
            support.sort()
            # Map the column indices of the local code to those of the support in the outer graph
            std_indices.append(support[lsc.indices])

            # Append the data
            std_data.append(lsc.data)

        # Apply cumulative sum to yield the overall indptr
        std_indptr = np.cumsum(np.concatenate(std_indptr))
        std_indices = np.concatenate(std_indices)
        std_data = np.concatenate(std_data)

        self.std_H = sp.csr_array((std_data, std_indices, std_indptr))

        

if __name__ == '__main__':
    cyclic_hamming = sp.csr_array([[1,0,1,0,1,0,1], 
                                   [0,1,1,0,1,1,0], 
                                   [0,0,0,1,1,1,1]])
    
    print(f'[n={cyclic_hamming.shape[1]}, k={code_dimension(cyclic_hamming)}, d={code_distance(cyclic_hamming)}]')
    
    # spoked_wheel = sp.csr_array([[1,1,1,1,1,1,1,0,0], 
                                #  [0,0,1,1,1,1,1,1,1]])
    K8 = nx.complete_graph(8)
    spoked_wheel = nx.Graph()
    spoked_wheel.add_nodes_from(K8.nodes, bipartite=0)
    spoked_wheel.add_nodes_from(K8.edges, bipartite=1)
    spoked_wheel.add_edges_from([(u, e) for u in K8.nodes for e in K8.edges if u in e])

    spkdwhl_H = bpt.biadjacency_matrix(spoked_wheel, row_order=K8.nodes, column_order=K8.edges)

    # The following commented code is based on ChatGPT's suggestion and yields a [28, 10, 6] code. 
    # sigma = npr.permutation(8)
    # indptr, indices, data = [0], [], []
    # for u in K8.nodes:
    #     count = 0
    #     for i, e in enumerate(K8.edges):
    #         if u in e:
    #             count += 1
    #             indices.append(i)
    #             x = np.unpackbits(np.uint8(sigma[e[0]]^sigma[e[1]]), count=3, bitorder='little').astype(np.int32).reshape(3, 1)
    #             data.append(x)

    #     indptr.append(count)
    # indptr = np.cumsum(indptr)
    # indices = np.array(indices)
    # data = np.stack(data, axis=0)

    # B = sp.csr_array(sp.bsr_array((data, indices, indptr)).todense())
    # print('outer matrix')
    # print(spkdwhl_H)
    # print(spkdwhl_H.todense())

    # print('full matrix')
    # print(B)
    # print(B.todense())
    # END OF [28, 10, 6] CONSTRUCTION

    order_assignment = { # [28, 6, 10]
        0: [0, 1, 2, 4, 3, 6, 7, 5], 
        1: [4, 0, 5, 6, 7, 3, 2, 1], 
        2: [4, 1, 0, 5, 6, 7, 3, 2], 
        3: [4, 2, 1, 0, 5, 6, 7, 3], 
        4: [4, 3, 2, 1, 0, 5, 6, 7], 
        5: [4, 7, 3, 2, 1, 0, 5, 6], 
        6: [4, 6, 7, 3, 2, 1, 0, 5], 
        7: [4, 5, 6, 7, 3, 2, 1, 0], 
    }

    # order_assignment = { # [28, 4, 13]
    #     k: np.roll(np.concatenate([np.array([0]), (1+np.roll(np.arange(7), shift=1))]), shift=k) for k in range(8)
    # }
    # order_assignment = { # [28, 5, 10]
    #     k: np.roll(np.concatenate([np.array([0]), (1+np.roll(np.arange(7), shift=2))]), shift=k) for k in range(8)
    # }
    # order_assignment = { # [28, 9, 6]
    #     k: np.roll(np.concatenate([np.array([0]), (1+np.roll(np.arange(7), shift=0))]), shift=k) for k in range(8)
    # }
    # w = npr.permutation(7)
    # order_assignment = { # [28, k, d]
    #     k: np.roll(np.concatenate([np.array([0]), (1+np.roll(w, shift=0))]), shift=k) for k in range(8)
    # }
    # order_assignment[u][v] is the column block
    
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

    B = sp.csr_array(sp.bsr_array((data, indices, indptr)).todense())
    print('outer matrix')
    print(spkdwhl_H.todense())

    print('full matrix')
    print(B.todense())
    print(B.sum())

    # order_assignments = [
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    #     np.array([], dtype=np.int32), 
    # ]
    
    print(f'[n={B.shape[1]}, k={code_dimension(B)}, d={code_distance(B)}]')
    print(f'[n={B.shape[0]}, k={code_dimension(B.T)}, d={code_distance(B.T)}]')
    

    # tanner_code = TannerCode(outer_graph=spkdwhl_H, local_subcode=cyclic_hamming, order_assignments=None)
    
    # for _ in range(100000):
    #     order_assignments = [npr.permutation(7) for _ in range(8)]
    #     tanner_code = TannerCode(outer_graph=spkdwhl_H, local_subcode=cyclic_hamming, order_assignments=order_assignments)
    #     if code_dimension(tanner_code.std_H) >= 9:
    #         break

    # print(tanner_code.std_H.todense())
    # print(f'[n={tanner_code.std_H.shape[1]}, k={code_dimension(tanner_code.std_H)}, d={code_distance(tanner_code.std_H)}]')



