# Test script for balanced configuration with current ground truth
# This tests the balanced config before running the full preprocessing

import sys
import os
import numpy as np
import time

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

# Temporarily replace the config module with balanced config
import sys
import config_balanced
sys.modules['config'] = config_balanced

# Import the environment (now uses balanced config)
from environment.constellation_env import ConstellationEnv

def test_balanced_configuration():
    """Test the balanced configuration with current ground truth."""
    print("=" * 60)
    print("TESTING BALANCED CONFIGURATION")
    print("=" * 60)
    
    # Print configuration summary
    print(f"Configuration Summary:")
    print(f"  Episodes: {config_balanced.TOTAL_EPISODES}")
    print(f"  Timesteps per episode: {config_balanced.SIMULATION_TIME_STEPS}")
    print(f"  Episode duration: {config_balanced.EPISODE_DURATION_SECONDS} seconds")
    print(f"  Number of satellites: {config_balanced.NUM_SATELLITES}")
    print(f"  Learning rate: {config_balanced.LEARNING_RATE}")
    print(f"  Batch size: {config_balanced.BATCH_SIZE}")
    print(f"  Hidden dimension: {config_balanced.HIDDEN_DIM}")
    print(f"  K nearest neighbors: {config_balanced.K_NEAREST_NEIGHBORS}")
    print()
    
    try:
        # Initialize the environment
        print("Initializing environment...")
        env = ConstellationEnv()
        
        # Test a single episode
        print("Testing single episode...")
        start_time = time.time()
        
        observation = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(config_balanced.SIMULATION_TIME_STEPS):
            # Generate random actions for testing
            actions = np.random.uniform(-1, 1, (config_balanced.NUM_SATELLITES, 2))
            
            # Take step
            observation, reward, done, info = env.step(actions)
            
            episode_reward += reward
            episode_steps += 1
            
            # Print progress every 10 steps
            if step % 10 == 0:
                print(f"  Step {step}/{config_balanced.SIMULATION_TIME_STEPS}: Reward = {reward:.3f}, "
                      f"Detection Rate = {info.get('detection_rate', 0):.3f}, "
                      f"OSPA = {info.get('ospa_distance', 0):.1f}")
            
            if done:
                break
        
        end_time = time.time()
        episode_time = end_time - start_time
        
        print(f"\nEpisode completed successfully!")
        print(f"  Total reward: {episode_reward:.3f}")
        print(f"  Steps completed: {episode_steps}")
        print(f"  Episode time: {episode_time:.2f} seconds")
        print(f"  Average step time: {episode_time/episode_steps:.3f} seconds")
        
        # Estimate full training time
        estimated_episode_time = episode_time * 1.1  # Add 10% for overhead
        estimated_training_time = estimated_episode_time * config_balanced.TOTAL_EPISODES / 3600  # Convert to hours
        
        print(f"\nTime Estimates:")
        print(f"  Estimated episode time: {estimated_episode_time:.2f} seconds")
        print(f"  Estimated training time: {estimated_training_time:.2f} hours")
        
        # Test environment properties
        print(f"\nEnvironment Properties:")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        print(f"  Ground truth objects per timestep: ~{len(env.ground_truth_database.get(0, []))}")
        
        print("\n✅ Balanced configuration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during balanced configuration test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ground_truth_loading():
    """Test that ground truth loads properly with balanced config."""
    print("\n" + "=" * 60)
    print("TESTING GROUND TRUTH LOADING")
    print("=" * 60)
    
    try:
        # Import ground truth loading
        from environment.constellation_env import ConstellationEnv
        
        # Initialize environment (this loads ground truth)
        env = ConstellationEnv()
        
        # Check ground truth properties
        gt_database = env.ground_truth_database
        
        print(f"Ground truth database loaded successfully!")
        print(f"  Number of timesteps: {len(gt_database)}")
        
        # Sample a few timesteps
        sample_timesteps = list(gt_database.keys())[:5]
        for timestep in sample_timesteps:
            objects = gt_database[timestep]
            print(f"  Timestep {timestep}: {len(objects)} objects")
            
            if objects:
                # Show sample object
                sample_obj = objects[0]
                print(f"    Sample object: ID={sample_obj['id']}, "
                      f"Position={sample_obj['position'][:3]}, "
                      f"Diameter={sample_obj['diameter']:.2f}")
        
        print("✅ Ground truth loading test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during ground truth loading test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function."""
    print("Starting balanced configuration tests...")
    
    # Test 1: Ground truth loading
    gt_success = test_ground_truth_loading()
    
    # Test 2: Environment functionality
    env_success = test_balanced_configuration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    if gt_success and env_success:
        print("✅ All tests passed! Balanced configuration is ready for training.")
        print("\nNext steps:")
        print("1. Run the 15-second preprocessing: python data_generation/preprocess_cpe_files_streaming_15s.py")
        print("2. Train with balanced config: python simulation/main.py (using config_balanced)")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        
        if not gt_success:
            print("  - Ground truth loading failed")
        if not env_success:
            print("  - Environment test failed")

if __name__ == "__main__":
    main() 