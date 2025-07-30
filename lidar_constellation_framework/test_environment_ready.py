#!/usr/bin/env python3
"""
Quick test to verify the environment is ready for training.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import simulation.config as config
from simulation.environment.constellation_env import ConstellationEnv
from simulation.rl_agent.ppo_agent import PPOAgent

def test_environment_ready():
    """Test that the environment can be created and run."""
    print("="*60)
    print("TESTING ENVIRONMENT READINESS")
    print("="*60)
    
    try:
        # Test environment creation
        print("Creating environment...")
        env = ConstellationEnv()
        print("✅ Environment created successfully")
        
        # Test environment reset
        print("Testing environment reset...")
        initial_state = env.reset()
        print(f"✅ Environment reset successful")
        print(f"   Initial state type: {type(initial_state)}")
        
        # Test agent creation
        print("Creating PPO agent...")
        agent = PPOAgent()
        print("✅ PPO agent created successfully")
        
        # Test action selection
        print("Testing action selection...")
        action, log_prob, value = agent.select_action(initial_state)
        print(f"✅ Action selection successful")
        print(f"   Action shape: {action.shape}")
        print(f"   Log prob: {log_prob}")
        print(f"   Value: {value}")
        
        # Test environment step
        print("Testing environment step...")
        next_state, reward, done, info = env.step(action)
        print(f"✅ Environment step successful")
        print(f"   Reward: {reward}")
        print(f"   Done: {done}")
        print(f"   Info keys: {list(info.keys())}")
        
        print("\n" + "="*60)
        print("🎉 ENVIRONMENT IS READY FOR TRAINING!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_environment_ready() 