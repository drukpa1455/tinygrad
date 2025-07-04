# PyTorch Geometric → tinygrad Implementation Plan

Based on exploration of the actual PyTorch Geometric codebase, here's a concrete implementation plan for recreating key functionality in tinygrad.

## Phase 1: Core Data Structure (FIRST STEP)

The foundation of PyG is the `Data` class. This should be our first implementation target.

### 1.1 Basic Data Class

Create `tinygrad/geometric/data.py`:

```python
from tinygrad import Tensor
from typing import Optional, Union, Any, Dict

class Data:
    """Graph data structure for tinygrad - simplified version of PyG's Data class"""
    
    def __init__(self, 
                 x: Optional[Tensor] = None,
                 edge_index: Optional[Tensor] = None,
                 edge_attr: Optional[Tensor] = None,
                 y: Optional[Union[Tensor, int, float]] = None,
                 **kwargs):
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
            return self.x.shape[0]
        elif self.edge_index is not None:
            return int(self.edge_index.max().numpy()) + 1
        return 0
    
    @property 
    def num_edges(self) -> int:
        """Number of edges in the graph"""
        if self.edge_index is not None:
            return self.edge_index.shape[1]
        return 0
    
    @property
    def num_node_features(self) -> int:
        """Number of node features"""
        if self.x is not None:
            return self.x.shape[1] if len(self.x.shape) > 1 else 1
        return 0
    
    def keys(self):
        """Return all attribute names"""
        return [key for key in self.__dict__.keys() if not key.startswith('_')]
    
    def __repr__(self):
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
```

**Why start here?** The Data class is used everywhere in PyG - it's the fundamental data structure that represents graphs.

### 1.2 Basic Utility Functions

Create `tinygrad/geometric/utils.py`:

```python
from tinygrad import Tensor
from typing import Tuple

def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """Add self-loops to the graph"""
    # Create self-loop edges [0,1,2,...,n-1]
    loop_edges = Tensor.arange(num_nodes).unsqueeze(0).repeat(2, 1)
    
    # Concatenate with existing edges
    return Tensor.cat([edge_index, loop_edges], dim=1)

def degree(index: Tensor, num_nodes: int) -> Tensor:
    """Compute node degrees"""
    # Count how many times each node appears in the index
    deg = Tensor.zeros(num_nodes)
    ones = Tensor.ones(index.shape[0])
    
    # This is a simplified scatter_add - we'll need proper implementation
    for i in range(index.shape[0]):
        node_idx = int(index[i].numpy())
        deg[node_idx] = deg[node_idx] + 1
    
    return deg
```

### 1.3 Test the Basic Data Structure

Create `test_basic_data.py`:

```python
from tinygrad import Tensor
from tinygrad.geometric.data import Data
from tinygrad.geometric.utils import add_self_loops, degree

# Create a simple graph: 0--1--2
edge_index = Tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # undirected edges
x = Tensor([[1.0], [2.0], [3.0]])  # node features
y = Tensor([0, 1, 0])  # node labels

# Create data object
data = Data(x=x, edge_index=edge_index, y=y)

print(f"Graph: {data}")
print(f"Num nodes: {data.num_nodes}")
print(f"Num edges: {data.num_edges}")  
print(f"Num features: {data.num_node_features}")

# Test utilities
edge_index_with_loops = add_self_loops(edge_index, data.num_nodes)
print(f"Edges with self-loops: {edge_index_with_loops.shape}")

degrees = degree(edge_index[0], data.num_nodes)  # count outgoing edges
print(f"Node degrees: {degrees}")
```

## Phase 2: Core Scatter Operations

**The biggest challenge**: PyG heavily relies on scatter operations that tinygrad doesn't have built-in.

### 2.1 Implement Scatter Operations

Create `tinygrad/geometric/scatter.py`:

```python
from tinygrad import Tensor

def scatter_add(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """
    Scatter add operation: result[index[i]] += src[i]
    
    This is critical for message aggregation in GNNs
    """
    # Initialize output tensor
    out = Tensor.zeros(dim_size, *src.shape[1:])
    
    # Manual implementation (slow but correct)
    # TODO: This needs to be optimized using tinygrad's backend
    for i in range(src.shape[0]):
        idx = int(index[i].numpy())
        out = out.cat([out[:idx], (out[idx] + src[i]).unsqueeze(0), out[idx+1:]], dim=0)
    
    return out

def scatter_mean(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Scatter mean operation"""
    summed = scatter_add(src, index, dim_size)
    count = scatter_add(Tensor.ones_like(index), index, dim_size)
    return summed / count.unsqueeze(1).maximum(Tensor([1.0]))

def scatter_max(src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Scatter max operation"""  
    out = Tensor.full((dim_size, *src.shape[1:]), float('-inf'))
    
    for i in range(src.shape[0]):
        idx = int(index[i].numpy())
        out[idx] = out[idx].maximum(src[i])
    
    return out
```

**Critical Note**: These scatter operations need to be optimized. The manual loop implementations above are just for getting started. We'll need to leverage tinygrad's efficient ops.

## Phase 3: Message Passing Framework

### 3.1 Base MessagePassing Class

Create `tinygrad/geometric/nn/message_passing.py`:

```python
from abc import ABC, abstractmethod
from tinygrad import Tensor
from typing import Dict, Any, Optional
from ..scatter import scatter_add, scatter_mean, scatter_max

class MessagePassing:
    """Base class for message passing layers - simplified version of PyG's MessagePassing"""
    
    def __init__(self, aggr: str = "add"):
        self.aggr = aggr
        self.flow = "source_to_target"  # Always source to target for simplicity
        
        # Aggregation functions
        self.aggr_functions = {
            "add": scatter_add,
            "mean": scatter_mean, 
            "max": scatter_max,
        }
    
    def propagate(self, edge_index: Tensor, x: Tensor, **kwargs) -> Tensor:
        """
        Main message passing propagation
        
        This is the core of PyG's framework:
        1. Create messages between connected nodes
        2. Aggregate messages at each node  
        3. Update node representations
        """
        # Get source and target node indices
        row, col = edge_index[0], edge_index[1]  # source, target
        
        # Lift node features to edge features
        x_i = x[col]  # target node features [num_edges, num_features]
        x_j = x[row]  # source node features [num_edges, num_features] 
        
        # Compute messages
        messages = self.message(x_i=x_i, x_j=x_j, **kwargs)
        
        # Aggregate messages at target nodes
        aggregated = self.aggregate(messages, col, dim_size=x.shape[0])
        
        # Update node representations  
        return self.update(aggregated, x=x, **kwargs)
    
    def message(self, x_i: Tensor, x_j: Tensor, **kwargs) -> Tensor:
        """Create messages from source nodes j to target nodes i"""
        return x_j  # Default: just pass source features
    
    def aggregate(self, inputs: Tensor, index: Tensor, dim_size: int) -> Tensor:
        """Aggregate messages at each node"""
        return self.aggr_functions[self.aggr](inputs, index, dim_size)
        
    def update(self, aggr_out: Tensor, **kwargs) -> Tensor:
        """Update node representations with aggregated messages"""
        return aggr_out  # Default: just return aggregated features
```

### 3.2 Simple GCN Layer

Create `tinygrad/geometric/nn/gcn_conv.py`:

```python
from tinygrad import Tensor, nn
from .message_passing import MessagePassing
from ..utils import add_self_loops, degree

class GCNConv(MessagePassing):
    """
    Graph Convolutional Network layer
    
    Based on: https://arxiv.org/abs/1609.02907
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(aggr="add")
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Linear transformation (this is the W matrix in GCN)
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        
    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        # Step 1: Add self-loops  
        edge_index = add_self_loops(edge_index, x.shape[0])
        
        # Step 2: Apply linear transformation
        x = self.lin(x)
        
        # Step 3: Compute normalization (degree-based)
        row, col = edge_index[0], edge_index[1]
        deg = degree(col, x.shape[0]).float()
        deg_inv_sqrt = deg.pow(-0.5)
        
        # Handle isolated nodes (degree = 0)
        deg_inv_sqrt = deg_inv_sqrt.masked_fill(deg_inv_sqrt == float('inf'), 0)
        
        # Compute normalization coefficients for each edge
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        
        # Step 4: Message passing with normalization
        return self.propagate(edge_index, x=x, norm=norm)
        
    def message(self, x_j: Tensor, norm: Tensor, **kwargs) -> Tensor:
        # Apply normalization to source node features
        return norm.unsqueeze(-1) * x_j
```

## Testing the Implementation

### Test Script

Create `test_gcn.py`:

```python
from tinygrad import Tensor
from tinygrad.geometric.data import Data
from tinygrad.geometric.nn.gcn_conv import GCNConv

# Create a simple 3-node graph
# 0 -- 1 -- 2 (line graph)
edge_index = Tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # undirected
x = Tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 2D node features

# Create GCN layer
gcn = GCNConv(in_channels=2, out_channels=4)

# Forward pass
out = gcn(x, edge_index)

print(f"Input shape: {x.shape}")
print(f"Output shape: {out.shape}")
print(f"Output:\n{out}")
```

## Next Steps (Phase 4+)

After getting the basic framework working:

1. **Optimize scatter operations** - This is critical for performance
2. **Add more layers**: GAT, SAGE, GIN
3. **Dataset loading**: Start with simple synthetic data, then Cora
4. **Training loop**: Implement a basic node classification example
5. **Batching**: Handle multiple graphs in a batch

## Key Implementation Challenges

1. **Scatter operations**: Need efficient implementation in tinygrad
2. **Memory management**: Graphs can be large and irregular
3. **Sparse tensors**: Many graph operations benefit from sparsity
4. **Gradient flow**: Ensuring backprop works through message passing

## Why This Approach?

1. **Start simple**: Data class is foundational but easy to implement
2. **Build incrementally**: Each component builds on the previous
3. **Test early**: Can verify correctness at each step
4. **Focus on core**: Message passing is the heart of all GNNs

The first concrete step is implementing the `Data` class and basic utilities, then testing with a simple graph. This gives us a foundation to build the message passing framework on top of.