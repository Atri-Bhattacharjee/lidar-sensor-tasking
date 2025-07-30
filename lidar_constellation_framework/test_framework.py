"""
Test script for LiDAR Constellation Sensor Tasking Framework

This script tests the basic functionality of all components to ensure
the framework is working correctly.
"""

import sys
import os
import numpy as np
import torch

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

def test_imports():
    """Test that all modules can be imported correctly."""
    print("Testing imports...")
    
    try:
        import config
        print("✓ config imported successfully")
        
        from environment.constellation_env import ConstellationEnv
        print("✓ ConstellationEnv imported successfully")
        
        from environment.perception_layer import PerceptionLayer
        print("✓ PerceptionLayer imported successfully")
        
        from environment.estimation_layer import EstimationLayer
        print("✓ EstimationLayer imported successfully")
        
        from rl_agent.gnn_model import ActorCriticGNN
        print("✓ ActorCriticGNN imported successfully")
        
        from rl_agent.ppo_agent import PPOAgent
        print("✓ PPOAgent imported successfully")
        
        from utils.ospa import calculate_ospa
        print("✓ OSPA utility imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_ospa():
    """Test OSPA distance calculation."""
    print("\nTesting OSPA calculation...")
    
    try:
        from utils.ospa import calculate_ospa
        
        # Test with empty sets
        ospa_empty = calculate_ospa([], [], cutoff_c=1000.0)
        assert ospa_empty == 0.0, f"Expected 0.0, got {ospa_empty}"
        print("✓ Empty sets OSPA calculation")
        
        # Test with single objects
        gt_obj = {'mean': np.array([100.0, 200.0, 300.0, 10.0, 20.0, 30.0])}
        est_obj = {'mean': np.array([110.0, 210.0, 310.0, 15.0, 25.0, 35.0])}
        
        ospa_single = calculate_ospa([gt_obj], [est_obj], cutoff_c=1000.0)
        assert ospa_single > 0.0, f"Expected positive OSPA, got {ospa_single}"
        print("✓ Single object OSPA calculation")
        
        return True
        
    except Exception as e:
        print(f"✗ OSPA test failed: {e}")
        return False


def test_perception_layer():
    """Test perception layer functionality."""
    print("\nTesting perception layer...")
    
    try:
        from environment.perception_layer import PerceptionLayer
        
        perception = PerceptionLayer()
        
        # Test measurement generation
        ground_truth = [
            {
                'id': 1,
                'position': [1000000.0, 2000000.0, 3000000.0],
                'velocity': [1000.0, 2000.0, 3000.0]
            }
        ]
        
        actions = np.random.uniform(-1, 1, (40, 2))  # 40 satellites, 2 angles each
        measurements = perception.generate_measurements(ground_truth, actions)
        
        assert isinstance(measurements, list), "Measurements should be a list"
        print("✓ Measurement generation")
        
        return True
        
    except Exception as e:
        print(f"✗ Perception layer test failed: {e}")
        return False


def test_estimation_layer():
    """Test estimation layer functionality."""
    print("\nTesting estimation layer...")
    
    try:
        from environment.estimation_layer import EstimationLayer
        
        estimation = EstimationLayer()
        
        # Test with empty measurements
        extracted_state, unassociated = estimation.step([], [], dt=1.0)
        assert isinstance(extracted_state, list), "Extracted state should be a list"
        assert isinstance(unassociated, list), "Unassociated should be a list"
        print("✓ Empty measurement processing")
        
        # Test with some measurements
        measurements = [
            {
                'range': 1000000.0,
                'azimuth': 45.0,
                'elevation': 30.0,
                'range_rate': 1000.0
            }
        ]
        
        extracted_state, unassociated = estimation.step(measurements, [], dt=1.0)
        print("✓ Measurement processing")
        
        return True
        
    except Exception as e:
        print(f"✗ Estimation layer test failed: {e}")
        return False


def test_gnn_model():
    """Test GNN model functionality."""
    print("\nTesting GNN model...")
    
    try:
        from rl_agent.gnn_model import ActorCriticGNN
        from torch_geometric.data import Data
        
        model = ActorCriticGNN()
        
        # Test with empty graph
        empty_data = Data(
            x=torch.zeros((0, 13), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            batch=torch.zeros(0, dtype=torch.long)
        )
        
        action_logits, state_value = model(empty_data)
        assert action_logits.shape == (1, 80), f"Expected shape (1, 80), got {action_logits.shape}"
        assert state_value.shape == (1, 1), f"Expected shape (1, 1), got {state_value.shape}"
        print("✓ Empty graph processing")
        
        # Test with single node
        single_node_data = Data(
            x=torch.randn((1, 13), dtype=torch.float32),
            edge_index=torch.zeros((2, 0), dtype=torch.long),
            batch=torch.zeros(1, dtype=torch.long)
        )
        
        action_logits, state_value = model(single_node_data)
        print("✓ Single node processing")
        
        return True
        
    except Exception as e:
        print(f"✗ GNN model test failed: {e}")
        return False


def test_environment():
    """Test environment functionality."""
    print("\nTesting environment...")
    
    try:
        from environment.constellation_env import ConstellationEnv
        
        env = ConstellationEnv()
        
        # Test reset
        initial_state = env.reset()
        assert hasattr(initial_state, 'x'), "State should have node features"
        assert hasattr(initial_state, 'edge_index'), "State should have edge index"
        print("✓ Environment reset")
        
        # Test step
        action = np.random.uniform(-1, 1, (40, 2))
        next_state, reward, done, info = env.step(action)
        
        assert isinstance(reward, float), "Reward should be float"
        assert isinstance(done, bool), "Done should be boolean"
        assert isinstance(info, dict), "Info should be dictionary"
        print("✓ Environment step")
        
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_ppo_agent():
    """Test PPO agent functionality."""
    print("\nTesting PPO agent...")
    
    try:
        from rl_agent.ppo_agent import PPOAgent
        from environment.constellation_env import ConstellationEnv
        
        agent = PPOAgent(device='cpu')
        env = ConstellationEnv()
        
        # Test action selection
        state = env.reset()
        action, log_prob, value = agent.select_action(state)
        
        assert action.shape == (40, 2), f"Expected action shape (40, 2), got {action.shape}"
        assert isinstance(log_prob, (float, np.ndarray)), "Log prob should be numeric"
        assert isinstance(value, (float, np.ndarray)), "Value should be numeric"
        print("✓ Action selection")
        
        # Test transition storage
        agent.store_transition(state, action, 0.0, log_prob, value, False)
        print("✓ Transition storage")
        
        return True
        
    except Exception as e:
        print(f"✗ PPO agent test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("LiDAR Constellation Framework - Test Suite")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_ospa,
        test_perception_layer,
        test_estimation_layer,
        test_gnn_model,
        test_environment,
        test_ppo_agent
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
        print("🎉 All tests passed! Framework is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 