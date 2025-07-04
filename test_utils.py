#!/usr/bin/env python3
"""
Test script for tinygrad.geometric utility functions.
"""

from tinygrad import Tensor
from tinygrad.geometric.utils import add_self_loops, degree, remove_self_loops, is_undirected, to_undirected

def test_add_self_loops():
    """Test adding self-loops to a graph"""
    print("Testing add_self_loops...")
    
    # Simple graph: 0-1, 1-2
    edge_index = Tensor([[0, 1], [1, 2]])
    num_nodes = 3
    
    result = add_self_loops(edge_index, num_nodes)
    print(f"Original edges: {edge_index.shape}")
    print(f"With self-loops: {result.shape}")
    print(f"Result:\n{result}")
    
    # Should have original edges plus self-loops
    assert result.shape == (2, 5)  # 2 original + 3 self-loops
    print("✓ add_self_loops test passed!")

def test_degree():
    """Test degree computation"""
    print("\nTesting degree...")
    
    # Graph: 0-1, 1-2, 1-2 (1 has degree 3, others have degree 1)
    edge_index = Tensor([[0, 1, 1], [1, 2, 2]])
    row = edge_index[0]  # source nodes
    num_nodes = 3
    
    deg = degree(row, num_nodes)
    print(f"Node degrees: {deg}")
    
    # Node 0: degree 1, Node 1: degree 2, Node 2: degree 0
    expected_degrees = [1, 2, 0]
    for i in range(num_nodes):
        assert abs(deg[i].item() - expected_degrees[i]) < 1e-6
    
    print("✓ degree test passed!")

def test_remove_self_loops():
    """Test removing self-loops"""
    print("\nTesting remove_self_loops...")
    
    # Graph with self-loops: 0-0, 0-1, 1-1, 1-2
    edge_index = Tensor([[0, 0, 1, 1], [0, 1, 1, 2]])
    
    result = remove_self_loops(edge_index)
    print(f"Original edges: {edge_index.shape}")
    print(f"Without self-loops: {result.shape}")
    print(f"Result:\n{result}")
    
    # Should only have 0-1 and 1-2 edges
    assert result.shape == (2, 2)
    print("✓ remove_self_loops test passed!")

def test_undirected():
    """Test undirected graph functions"""
    print("\nTesting undirected graph functions...")
    
    # Directed graph: 0->1, 1->2
    directed_edge_index = Tensor([[0, 1], [1, 2]])
    
    # Check if undirected
    is_undir = is_undirected(directed_edge_index)
    print(f"Is directed graph undirected? {is_undir}")
    assert not is_undir
    
    # Convert to undirected
    undirected_edge_index = to_undirected(directed_edge_index)
    print(f"Undirected edges: {undirected_edge_index.shape}")
    print(f"Result:\n{undirected_edge_index}")
    
    # Check if result is undirected
    is_undir_result = is_undirected(undirected_edge_index)
    print(f"Is result undirected? {is_undir_result}")
    assert is_undir_result
    
    print("✓ undirected graph tests passed!")

if __name__ == "__main__":
    test_add_self_loops()
    test_degree()
    test_remove_self_loops()
    test_undirected()
    print("\n🎉 All utility tests passed! Ready for message passing.")