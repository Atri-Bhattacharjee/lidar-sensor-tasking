#!/usr/bin/env python3
"""
Test script to verify all imports work correctly.
"""

import sys
import os

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

try:
    print("Testing imports...")
    
    # Test config import
    import config
    print("✅ Config imported successfully")
    
    # Test environment imports
    from environment.constellation_env import ConstellationEnv
    print("✅ ConstellationEnv imported successfully")
    
    from environment.perception_layer import PerceptionLayer
    print("✅ PerceptionLayer imported successfully")
    
    from environment.estimation_layer import EstimationLayer
    print("✅ EstimationLayer imported successfully")
    
    # Test RL agent imports
    from rl_agent.ppo_agent import PPOAgent
    print("✅ PPOAgent imported successfully")
    
    from rl_agent.gnn_model import ActorCriticGNN
    print("✅ ActorCriticGNN imported successfully")
    
    # Test utils imports
    from utils.ospa import calculate_ospa
    print("✅ OSPA function imported successfully")
    
    from utils.performance_monitor import PerformanceMonitor
    print("✅ PerformanceMonitor imported successfully")
    
    print("\n🎉 ALL IMPORTS SUCCESSFUL!")
    print("The system is ready for training.")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    import traceback
    traceback.print_exc() 