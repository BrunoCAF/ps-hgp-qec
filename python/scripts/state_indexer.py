import numpy as np
import networkx as nx
import networkx.algorithms.bipartite as bpt
import networkx.algorithms.isomorphism as iso
import scipy.sparse as sp

class StateIndexer:
    """
    The purpose of this class is to map states (represented by their Tanner 
    graphs) to row indices in the PS matrices. The interface is concentrated
    in the get_index() method, that takes a MultiGraph object and returns an
    integer index in [0..S-1] where S is the current number of known states. 
    """
    def __init__(self, graph_dims: tuple[int, int], serialization='sparse'):
        """
        Initialize the StateIndexer.

        :param graph_dims: The shape of the dense biadjacency matrix representing the graph.
        :param serialization: The serialization method to be employed, either 'sparse' or 'dense'.
        """
        self.graph_dims = graph_dims
        serialization_methods = {'sparse': (self.serialize_sparse, self.deserialize_sparse), 
                                 'dense': (self.serialize_dense, self.deserialize_dense)}
        self.serialize, self.deserialize = serialization_methods[serialization]

        self.storage = {}
        self.next_index = 0

    def get_index(self, G: nx.MultiGraph) -> int:
        """
        Retrieve the PS index associated with the state represented by G.

        :param G: A MultiGraph object representing the state.
        :return: The index associated with the state.
        """
        skey = self.serialize(G)
        if skey in self.storage:
            return self.storage[skey]
        else:
            self.storage[skey] = self.next
            self.next += 1
        return self.next - 1

    def get_index_simple(self, G: nx.MultiGraph) -> int:
        """
        Retrieve the reward cache index associated with the state represented by G.

        :param G: A MultiGraph object representing the state.
        :return: The index associated with the state.
        """
        return self.get_index(nx.Graph(G))

    def serialize_sparse(self, G: nx.MultiGraph) -> bytes:
        """
        Serialize the biadjacency matrix from its sparse CSR format. 
        Since the shape of the matrix is known, dump only indptr, indices, and data.

        :param G: A MultiGraph object representing the state.
        :return: The serialized biadjacency matrix as bytes.
        """
        m = self.graph_dims[0]
        H = bpt.biadjacency_matrix(G, row_order=np.arange(m)).astype(np.uint8)
        std_serial = lambda x: x.astype(np.uint8).tobytes()
        return std_serial(H.indptr) + std_serial(H.indices) + std_serial(H.data)

    def deserialize_sparse(self, g: bytes) -> nx.MultiGraph:
        """
        Deserialize the biadjacency matrix into its sparse CSR format.

        :param g: The serialized biadjacency matrix as bytes.
        :return: A MultiGraph object.
        """
        indptr_len = self.graph_dims[0] + 1
        data_len = (len(g) - indptr_len)//2
        
        indptr = np.frombuffer(g[:indptr_len], dtype=np.uint8)
        indices = np.frombuffer(g[indptr_len:][:data_len], dtype=np.uint8)
        data = np.frombuffer(g[indptr_len:][data_len:], dtype=np.uint8)
        
        H = sp.csr_matrix((data, indices, indptr), shape=self.graph_dims)
        return bpt.from_biadjacency_matrix(H, create_using=nx.MultiGraph)

    def serialize_dense(self, G: nx.MultiGraph) -> bytes:
        """
        Serialize the graph into a bytestring from its dense biadjacency matrix.

        :param G: A MultiGraph object representing the state.
        :return: The serialized biadjacency matrix as bytes.
        """
        m = self.graph_dims[0]
        H = bpt.biadjacency_matrix(G, row_order=np.arange(m)).astype(np.uint8).todense()
        return H.tobytes()

    def deserialize_dense(self, g: bytes) -> nx.MultiGraph:
        """
        Deserialize a bytestring into a dense biadjacency matrix.

        :param g: The serialized biadjacency matrix as bytes.
        :return: A MultiGraph object.
        """
        H = np.frombuffer(g, dtype=np.uint8).reshape(self.graph_dims)
        return bpt.from_biadjacency_matrix(sp.csr_matrix(H), create_using=nx.MultiGraph)
        

class IsomorphicStateIndexer(StateIndexer):
    """
    Specialized indexer where states with isomorphic graphs will be mapped to 
    the same index, when the number of isomorphic duplicates becomes harmful 
    in terms of memory and reward sampling cost. 
    """
    def __init__(self, graph_dims: tuple[int, int], serialization='sparse', 
                 hash_params: dict={'iterations': 20, 'digest_size': 32}, debug=False):
        """
        Initialize the IsomorphicStateIndexer.

        :param graph_dims: The shape of the dense biadjacency matrix representing the graph.
        :param serialization: The serialization method to be employed, either 'sparse' or 'dense'.
        :param hash_params: Parameters to the WL graph hash method.
        """
        super().__init__(graph_dims, serialization)
        self.hash_params = hash_params
        
        self.debug = debug
        if self.debug:
            self.collisions = {}

    def get_index(self, G: nx.MultiGraph) -> int:
        """
        Retrieve the PS index associated with the state represented by G.

        :param G: A MultiGraph object representing the state.
        :return: The index associated with the state.
        """
        G_hkey = self.graph_hash(G)
        G_skey = self.serialize(G)

        if G_hkey in self.storage:
            
            if self.debug:
                if G_hkey not in self.collisions:
                    self.collisions[G_hkey] = {'h': 1, 's': 0, 'f': 0, 'i': 0}
                else:
                    self.collisions[G_hkey]['h'] += 1
            
            for skey in self.storage[G_hkey]:
                if self.debug:
                    self.collisions[G_hkey]['s'] += 1
                if skey == G_skey:
                    return self.storage[G_hkey][skey]
                
                G_old = self.deserialize(skey)
                
                if self.debug:
                    self.collisions[G_hkey]['f'] += 1
                if iso.faster_could_be_isomorphic(G, G_old):

                    if self.debug:
                        self.collisions[G_hkey]['i'] += 1
                    if iso.vf2pp_is_isomorphic(G, G_old):
                        return self.storage[G_hkey][skey]
            
            self.storage[G_hkey][G_skey] = self.next_index
        else:
            self.storage[G_hkey] = {G_skey: self.next_index}
        
        self.next_index += 1
        
        if self.debug:
            print(self.next_index - 1)
        
        return self.next_index - 1

    def collision_report(self):
        assert self.debug
        print(self.collisions)
        print([len(s_dict) for s_dict in self.storage.values()])

    def graph_hash(self, G: nx.MultiGraph) -> str:
        """
        Get the Weisfeiler-Lehman graph hash.

        :param G: A MultiGraph object representing the state.
        :return: The WL graph hash as a string.
        """
        return nx.algorithms.weisfeiler_lehman_graph_hash(G, **self.hash_params)

