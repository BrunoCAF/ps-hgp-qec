import numpy as np
import networkx as nx
import networkx.algorithms.bipartite as bpt
import scipy.sparse as sp


class GraphSerializer:
    def __init__(self, graph_dims: tuple[int, int], serialization='sparse'):
        """
        Initialize the GraphSerializer.

        :param graph_dims: The shape of the dense biadjacency matrix representing the graph.
        :param serialization: The serialization method to be employed, 'sparse'|'dense'.
        """
        self.graph_dims = graph_dims
        serialization_methods = {'sparse': self.serialize_sparse, 
                                 'dense': self.serialize_dense,
                                }
        self.serialize = serialization_methods[serialization]

    def serialize_sparse(self, G: nx.MultiGraph) -> bytes:
        """
        Serialize the list of edges of the graph.

        :param G: A MultiGraph object representing the state.
        :return: The serialized edge list as bytes.
        """
        return np.array(sorted(G.edges(data=False)), dtype=np.uint8).flatten().tobytes()

    def serialize_dense(self, G: nx.MultiGraph) -> bytes:
        """
        Serialize the graph into a bytestring from its dense biadjacency matrix.

        :param G: A MultiGraph object representing the state.
        :return: The serialized biadjacency matrix as bytes.
        """
        m, n = self.graph_dims
        H = bpt.biadjacency_matrix(G, row_order=np.arange(m), column_order=np.arange(m, m+n)).astype(np.uint8).todense()
        return H.tobytes()
           
    def deserialize_sparse(self, g: bytes) -> nx.MultiGraph:
        """
        Deserialize the edge list back into the graph.

        :param g: The serialized edge list as bytes.
        :return: A MultiGraph object.
        """
        m, n = self.graph_dims
        edgelist = np.frombuffer(g, dtype=np.uint8)
        
        G = nx.MultiGraph()
        G.add_nodes_from(np.arange(m), bipartite=0)
        G.add_nodes_from(np.arange(m, m+n), bipartite=1)
        G.add_edges_from([tuple(r) for r in edgelist.reshape(-1, 2)])
        
        return G

    def deserialize_dense(self, g: bytes) -> nx.MultiGraph:
        """
        Deserialize a bytestring into a dense biadjacency matrix.

        :param g: The serialized biadjacency matrix as bytes.
        :return: A MultiGraph object.
        """
        H = np.frombuffer(g, dtype=np.uint8).reshape(self.graph_dims)
        return bpt.from_biadjacency_matrix(sp.csr_matrix(H), create_using=nx.MultiGraph)
    

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
        self.GS = GraphSerializer(graph_dims, serialization)
        
        self.storage = {}
        self.next_index = 0

    def get_index(self, G: nx.MultiGraph) -> int:
        """
        Retrieve the PS index associated with the state represented by G.

        :param G: A MultiGraph object representing the state.
        :return: The index associated with the state.
        """
        skey = self.GS.serialize(G)
        if skey in self.storage:
            return self.storage[skey]
        else:
            self.storage[skey] = self.next_index
            self.next_index += 1
        return self.next_index - 1
    
    def invert_index(self) -> dict[int, bytes]:
        return {i: sk for sk, i in self.storage.items()}

        

class RewardCache:
    """
    The purpose of this class is to map states (represented by their Tanner 
    graphs) to a dictionary containing the previously computed rewards. Works
    exactly like StateIndexer, but MultiGraphs are converted to simple Graphs
    before computing the reward, so the cache index may differ from the state
    index. 
    """
    def __init__(self, graph_dims: tuple[int, int], serialization='sparse'):
        self.GS = GraphSerializer(graph_dims, serialization)
        self.cache = {}

    def __contains__(self, G: nx.MultiGraph) -> bool:
        return self.GS.serialize(nx.Graph(G)) in self.cache
    
    def __getitem__(self, G: nx.MultiGraph) -> float:
        return self.cache[self.GS.serialize(nx.Graph(G))]
    
    def __setitem__(self, G: nx.MultiGraph, value: float):
        self.cache[self.GS.serialize(nx.Graph(G))] = value
