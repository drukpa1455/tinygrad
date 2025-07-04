#!/usr/bin/env python3
"""
Test script for basic tinygrad.geometric Data class functionality.

This is the first step in implementing PyTorch Geometric functionality in tinygrad.
"""

from tinygrad import Tensor
from tinygrad.geometric.data import Data

def test_basic_data():
    """Test basic Data class functionality"""
    print("Testing tinygrad.geometric.Data class...")
    
    # Create a simple graph: 0--1--2 (line graph)
    edge_index = Tensor([[0, 1, 1, 2], [1, 0, 2, 1]])  # undirected edges
    x = Tensor([[1.0], [2.0], [3.0]])  # node features
    y = Tensor([0, 1, 0])  # node labels

    # Create data object
    data = Data(x=x, edge_index=edge_index, y=y)

    print(f"Graph: {data}")
    print(f"Num nodes: {data.num_nodes}")
    print(f"Num edges: {data.num_edges}")  
    print(f"Num features: {data.num_node_features}")
    
    # Test dictionary-like access
    print(f"Keys: {data.keys()}")
    print(f"Has 'x': {'x' in data}")
    print(f"Has 'z': {'z' in data}")
    
    # Test accessing attributes
    print(f"Node features shape: {data['x'].shape}")
    print(f"Edge index shape: {data['edge_index'].shape}")
    
    print("✓ Basic Data class test passed!")

def test_empty_data():
    """Test Data class with minimal data"""
    print("\nTesting empty Data object...")
    
    # Create empty data object
    data = Data()
    
    print(f"Empty graph: {data}")
    print(f"Num nodes: {data.num_nodes}")
    print(f"Num edges: {data.num_edges}")
    print(f"Num features: {data.num_node_features}")
    
    print("✓ Empty Data test passed!")

def test_edge_only_data():
    """Test Data class with only edges (no node features)"""
    print("\nTesting Data with only edges...")
    
    # Create data with only edge_index
    edge_index = Tensor([[0, 1, 2], [1, 2, 0]])  # triangle graph
    data = Data(edge_index=edge_index)
    
    print(f"Edge-only graph: {data}")
    print(f"Num nodes: {data.num_nodes}")  # Should infer from edge_index
    print(f"Num edges: {data.num_edges}")
    print(f"Num features: {data.num_node_features}")  # Should be 0
    
    print("✓ Edge-only Data test passed!")

if __name__ == "__main__":
    test_basic_data()
    test_empty_data()
    test_edge_only_data()
    print("\n🎉 All basic Data tests passed! Ready for next step.")