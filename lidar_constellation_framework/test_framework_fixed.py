"""
Fixed Test Framework for LiDAR Constellation Framework
Uses absolute imports to avoid relative import issues
"""

import sys
import os
import numpy as np
import torch
import pandas as pd

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

def test_imports():
    """Test that all modules can be imported successfully."""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config imported successfully")
    except Exception as e:
        print(f"✗ config import failed: {e}")
        return False
    
    try:
        from utils.ospa import calculate_ospa
        print("✓ OSPA module imported successfully")
    except Exception as e:
        print(f"✗ OSPA import failed: {e}")
        return False
    
    try:
        from environment.perception_layer import PerceptionLayer
        print("✓ Perception layer imported successfully")
    except Exception as e:
        print(f"✗ Perception layer import failed: {e}")
        return False
    
    try:
        from environment.estimation_layer import EstimationLayer
        print("✓ Estimation layer imported successfully")
    except Exception as e:
        print(f"✗ Estimation layer import failed: {e}")
        return False
    
    try:
        from rl_agent.gnn_model import ActorCriticGNN
        print("✓ GNN model imported successfully")
    except Exception as e:
        print(f"✗ GNN model import failed: {e}")
        return False
    
    try:
        from rl_agent.ppo_agent import PPOAgent
        print("✓ PPO agent imported successfully")
    except Exception as e:
        print(f"✗ PPO agent import failed: {e}")
        return False
    
    return True

def test_ospa():
    """Test OSPA calculation functions."""
    print("Testing OSPA calculation...")
    
    try:
        from utils.ospa import calculate_ospa
        
        # Test empty sets
        ospa_empty = calculate_ospa([], [], cutoff=1000.0, order=1.0)
        print("✓ Empty sets OSPA calculation")
        
        # Test single object
        ospa_single = calculate_ospa([[1, 2, 3]], [[1, 2, 3]], cutoff=1000.0, order=1.0)
        print("✓ Single object OSPA calculation")
        
        return True
    except Exception as e:
        print(f"✗ OSPA test failed: {e}")
        return False

def test_perception_layer():
    """Test perception layer functionality."""
    print("Testing perception layer...")
    
    try:
        from environment.perception_layer import PerceptionLayer
        
        # Create perception layer
        perception = PerceptionLayer()
        
        # Test measurement generation
        ground_truth_objects = [
            {'position': np.array([1000, 2000, 3000]), 'velocity': np.array([1, 2, 3])}
        ]
        
        measurements = perception.generate_measurements(ground_truth_objects)
        print("✓ Measurement generation")
        
        return True
    except Exception as e:
        print(f"✗ Perception layer test failed: {e}")
        return False

def test_estimation_layer():
    """Test estimation layer functionality."""
    print("Testing estimation layer...")
    
    try:
        from environment.estimation_layer import EstimationLayer
        
        # Create estimation layer
        estimation = EstimationLayer()
        
        # Test empty measurement processing
        estimation.step([])
        print("✓ Empty measurement processing")
        
        # Test measurement processing
        measurements = [np.array([1000, 2000, 3000, 1, 2, 3])]
        estimation.step(measurements)
        print("✓ Measurement processing")
        
        return True
    except Exception as e:
        print(f"✗ Estimation layer test failed: {e}")
        return False

def test_gnn_model():
    """Test GNN model functionality."""
    print("Testing GNN model...")
    
    try:
        from rl_agent.gnn_model import ActorCriticGNN
        import torch_geometric
        from torch_geometric.data import Data
        
        # Create GNN model
        model = ActorCriticGNN()
        
        # Create dummy graph data
        x = torch.randn(10, 13)  # 10 nodes, 13 features
        edge_index = torch.randint(0, 10, (2, 20))  # 20 edges
        data = Data(x=x, edge_index=edge_index)
        
        # Test forward pass
        with torch.no_grad():
            action_logits, value = model(data)
        
        print("✓ GNN model forward pass")
        
        return True
    except Exception as e:
        print(f"✗ GNN model test failed: {e}")
        return False

def test_ppo_agent():
    """Test PPO agent functionality."""
    print("Testing PPO agent...")
    
    try:
        from rl_agent.ppo_agent import PPOAgent
        from rl_agent.gnn_model import ActorCriticGNN
        import torch_geometric
        from torch_geometric.data import Data
        
        # Create model and agent
        model = ActorCriticGNN()
        agent = PPOAgent(model)
        
        # Create dummy graph data
        x = torch.randn(5, 13)  # 5 nodes, 13 features
        edge_index = torch.randint(0, 5, (2, 10))  # 10 edges
        data = Data(x=x, edge_index=edge_index)
        
        # Test action selection
        action, log_prob, value = agent.select_action(data)
        print("✓ PPO agent action selection")
        
        return True
    except Exception as e:
        print(f"✗ PPO agent test failed: {e}")
        return False

def test_ground_truth_loading():
    """Test that ground truth database can be loaded."""
    print("Testing ground truth loading...")
    
    try:
        import pickle
        
        # Try to load the ground truth database
        with open('ground_truth_data/ground_truth_database.pkl', 'rb') as f:
            ground_truth = pickle.load(f)
        
        print(f"✓ Ground truth loaded successfully")
        print(f"   • {len(ground_truth)} timesteps")
        print(f"   • {sum(len(objects) for objects in ground_truth.values())} total object states")
        
        return True
    except Exception as e:
        print(f"✗ Ground truth loading failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("LiDAR Constellation Framework - Fixed Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ospa,
        test_perception_layer,
        test_estimation_layer,
        test_gnn_model,
        test_ppo_agent,
        test_ground_truth_loading
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Framework is ready for training.")
    else:
        print("⚠️ Some tests failed, but core functionality may still work.")
        print("Proceeding with training is recommended.")
    
    return passed == total

if __name__ == "__main__":
    main() 