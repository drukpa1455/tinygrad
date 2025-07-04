"""
Basic utility functions for graph operations in tinygrad.geometric
"""

from tinygrad import Tensor
from typing import Tuple


def add_self_loops(edge_index: Tensor, num_nodes: int) -> Tensor:
    """
    Add self-loops to the graph.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        num_nodes: Number of nodes in the graph
        
    Returns:
        New edge_index with self-loops added [2, num_edges + num_nodes]
    """
    # Create self-loop edges [0,1,2,...,n-1] -> [0,1,2,...,n-1]
    loop_index = Tensor.arange(num_nodes)
    loop_edges = loop_index.unsqueeze(0).repeat(2, 1)
    
    # Concatenate with existing edges
    return edge_index.cat(loop_edges, dim=1)


def degree(index: Tensor, num_nodes: int) -> Tensor:
    """
    Compute node degrees by counting occurrences in index.
    
    This is a simplified version - for full implementation we need proper scatter operations.
    
    Args:
        index: Node indices to count [num_edges]
        num_nodes: Total number of nodes
        
    Returns:
        Degree of each node [num_nodes]
    """
    # Initialize degree tensor
    deg = Tensor.zeros(num_nodes)
    
    # Manual counting (inefficient but correct)
    # TODO: Replace with efficient scatter_add when available
    for i in range(int(index.shape[0])):
        node_idx = int(index[i].item())
        if 0 <= node_idx < num_nodes:
            # This is very inefficient but works for small graphs
            deg_val = deg[node_idx].item() + 1
            # Create new tensor with updated value
            deg_list = [deg[j].item() for j in range(num_nodes)]
            deg_list[node_idx] = deg_val
            deg = Tensor(deg_list)
    
    return deg


def remove_self_loops(edge_index: Tensor) -> Tensor:
    """
    Remove self-loops from edge_index.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        
    Returns:
        Edge indices without self-loops
    """
    # Find edges that are not self-loops
    row, col = edge_index[0], edge_index[1]
    mask = row != col
    
    # This is a bit tricky in tinygrad - we need to filter
    # For now, let's use a simple approach
    non_self_loop_indices = []
    for i in range(int(edge_index.shape[1])):
        if row[i].item() != col[i].item():
            non_self_loop_indices.append(i)
    
    if not non_self_loop_indices:
        # No edges left, return empty tensor
        return Tensor.zeros(2, 0, dtype=edge_index.dtype)
    
    # Select non-self-loop edges
    filtered_edges = []
    for i in non_self_loop_indices:
        filtered_edges.append([edge_index[0, i].item(), edge_index[1, i].item()])
    
    return Tensor(filtered_edges).T


def is_undirected(edge_index: Tensor) -> bool:
    """
    Check if the graph is undirected (every edge has a reverse edge).
    
    Args:
        edge_index: Edge indices [2, num_edges]
        
    Returns:
        True if graph is undirected
    """
    # Convert to edge sets for easier comparison
    edges = set()
    reverse_edges = set()
    
    for i in range(int(edge_index.shape[1])):
        src, dst = int(edge_index[0, i].item()), int(edge_index[1, i].item())
        edges.add((src, dst))
        reverse_edges.add((dst, src))
    
    return edges == reverse_edges


def to_undirected(edge_index: Tensor) -> Tensor:
    """
    Convert a directed graph to undirected by adding reverse edges.
    
    Args:
        edge_index: Edge indices [2, num_edges]
        
    Returns:
        Undirected edge indices [2, num_edges']
    """
    # Create reverse edges  
    reverse_edge_index = edge_index[[1, 0], :]  # Swap rows
    
    # Concatenate original and reverse edges
    undirected_edges = edge_index.cat(reverse_edge_index, dim=1)
    
    # Remove duplicates (this is simplified - full implementation would be more efficient)
    edges_set = set()
    unique_edges = []
    
    for i in range(int(undirected_edges.shape[1])):
        src, dst = int(undirected_edges[0, i].item()), int(undirected_edges[1, i].item())
        edge = tuple(sorted([src, dst]))  # Canonical form
        if edge not in edges_set:
            edges_set.add(edge)
            unique_edges.append([src, dst])
            if src != dst:  # Add reverse if not self-loop
                unique_edges.append([dst, src])
    
    if not unique_edges:
        return Tensor.zeros(2, 0, dtype=edge_index.dtype)
    
    return Tensor(unique_edges).T