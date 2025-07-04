"""
Graph data structures for tinygrad.geometric
"""

from tinygrad import Tensor
from typing import Optional, Union, Any, List


class Data:
    """
    Graph data structure for tinygrad - simplified version of PyG's Data class
    
    A data object describing a homogeneous graph.
    The data object can hold node-level, edge-level and graph-level attributes.
    """
    
    def __init__(self, 
                 x: Optional[Tensor] = None,
                 edge_index: Optional[Tensor] = None,
                 edge_attr: Optional[Tensor] = None,
                 y: Optional[Union[Tensor, int, float]] = None,
                 **kwargs):
        """
        Initialize a graph data object.
        
        Args:
            x: Node features [num_nodes, num_node_features]
            edge_index: Edge indices [2, num_edges] in COO format
            edge_attr: Edge features [num_edges, num_edge_features]
            y: Graph or node labels/targets
            **kwargs: Additional attributes
        """
        # Core attributes 
        self.x = x                    # Node features [num_nodes, num_features]
        self.edge_index = edge_index  # Edge indices [2, num_edges] 
        self.edge_attr = edge_attr    # Edge features [num_edges, num_edge_features]
        self.y = y                    # Labels/targets
        
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the graph"""
        if self.x is not None:
            return int(self.x.shape[0])
        elif self.edge_index is not None:
            return int(self.edge_index.max().item()) + 1
        return 0
    
    @property 
    def num_edges(self) -> int:
        """Number of edges in the graph"""
        if self.edge_index is not None:
            return int(self.edge_index.shape[1])
        return 0
    
    @property
    def num_node_features(self) -> int:
        """Number of node features"""
        if self.x is not None:
            return int(self.x.shape[1]) if len(self.x.shape) > 1 else 1
        return 0
    
    def keys(self) -> List[str]:
        """Return all attribute names"""
        return [key for key in self.__dict__.keys() if not key.startswith('_')]
    
    def __repr__(self) -> str:
        """String representation of the data object"""
        info = []
        if self.x is not None:
            info.append(f"x={list(self.x.shape)}")
        if self.edge_index is not None:
            info.append(f"edge_index={list(self.edge_index.shape)}")
        if self.edge_attr is not None:
            info.append(f"edge_attr={list(self.edge_attr.shape)}")
        if self.y is not None:
            if isinstance(self.y, Tensor):
                info.append(f"y={list(self.y.shape)}")
            else:
                info.append(f"y={self.y}")
        
        return f"Data({', '.join(info)})"
    
    def __getitem__(self, key: str) -> Any:
        """Access attributes like a dictionary"""
        return getattr(self, key)
    
    def __setitem__(self, key: str, value: Any):
        """Set attributes like a dictionary"""
        setattr(self, key, value)
    
    def __contains__(self, key: str) -> bool:
        """Check if attribute exists"""
        return hasattr(self, key)