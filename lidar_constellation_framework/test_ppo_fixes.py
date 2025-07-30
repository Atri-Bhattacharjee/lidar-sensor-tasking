#!/usr/bin/env python3
"""
Test script to verify the PPO agent fixes work correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
from torch_geometric.data import Data

# Import config variables
import simulation.config as config
from simulation.rl_agent.gnn_model import ActorCriticGNN
from simulation.rl_agent.ppo_agent import PPOAgent

def test_gnn_model():
    """Test the GNN model output shapes."""
    print("Testing GNN model...")
    
    # Create a dummy graph
    num_nodes = 10
    x = torch.randn(num_nodes, 13)  # Node features
    edge_index = torch.randint(0, num_nodes, (2, 20))  # Random edges
    batch = torch.zeros(num_nodes, dtype=torch.long)  # All nodes in batch 0
    
    graph_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Create model
    model = ActorCriticGNN()
    
    # Forward pass
    action_logits, state_value = model(graph_data)
    
    print(f"✅ GNN Model Test:")
    print(f"   Action logits shape: {action_logits.shape}")
    print(f"   State value shape: {state_value.shape}")
    print(f"   Expected action shape: (1, {config.NUM_SATELLITES}, 2)")
    print(f"   Expected value shape: (1, 1)")
    
    assert action_logits.shape == (1, config.NUM_SATELLITES, 2), f"Wrong action shape: {action_logits.shape}"
    assert state_value.shape == (1, 1), f"Wrong value shape: {state_value.shape}"
    
    return True

def test_ppo_agent():
    """Test the PPO agent action selection."""
    print("\nTesting PPO agent...")
    
    # Create agent
    agent = PPOAgent()
    
    # Create a dummy graph
    num_nodes = 10
    x = torch.randn(num_nodes, 13)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    
    graph_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    # Select action
    action, log_prob, value = agent.select_action(graph_data)
    
    print(f"✅ PPO Agent Test:")
    print(f"   Action shape: {action.shape}")
    print(f"   Log prob shape: {log_prob.shape}")
    print(f"   Value shape: {value.shape}")
    print(f"   Expected action shape: ({config.NUM_SATELLITES}, 2)")
    print(f"   Expected log prob shape: ()")
    print(f"   Expected value shape: ()")
    
    assert action.shape == (config.NUM_SATELLITES, 2), f"Wrong action shape: {action.shape}"
    assert log_prob.shape == (), f"Wrong log prob shape: {log_prob.shape}"
    assert value.shape == (), f"Wrong value shape: {value.shape}"
    
    return True

def test_ppo_memory():
    """Test the PPO memory buffer."""
    print("\nTesting PPO memory...")
    
    # Create agent
    agent = PPOAgent()
    
    # Create dummy data
    num_nodes = 10
    x = torch.randn(num_nodes, 13)
    edge_index = torch.randint(0, num_nodes, (2, 20))
    batch = torch.zeros(num_nodes, dtype=torch.long)
    graph_data = Data(x=x, edge_index=edge_index, batch=batch)
    
    action = np.random.randn(config.NUM_SATELLITES, 2)
    reward = 1.0
    log_prob = 0.5
    value = 0.8
    done = False
    
    # Store transition
    agent.store_transition(graph_data, action, reward, log_prob, value, done)
    
    print(f"✅ PPO Memory Test:")
    print(f"   Memory length: {len(agent.memory)}")
    print(f"   Expected: 1")
    
    assert len(agent.memory) == 1, f"Memory length wrong: {len(agent.memory)}"
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing PPO Agent Fixes")
    print("=" * 60)
    
    try:
        test_gnn_model()
        test_ppo_agent()
        test_ppo_memory()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS PASSED!")
        print("The PPO agent fixes are working correctly.")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 