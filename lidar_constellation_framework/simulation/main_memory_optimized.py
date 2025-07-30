"""
Memory-Optimized Main Training Script for LiDAR Constellation Sensor Tasking

This script orchestrates the complete training process with memory optimization
to prevent memory issues during training.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
import random
import psutil
import gc

# Add the simulation directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment.constellation_env import ConstellationEnv
from rl_agent.ppo_agent_memory_optimized import MemoryOptimizedPPOAgent
from utils.performance_monitor import PerformanceMonitor


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_memory_usage():
    """
    Get current memory usage information.
    
    Returns:
        Dictionary with memory usage information
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }


def plot_training_curves(training_stats: Dict[str, List[float]], save_path: str):
    """
    Plot training curves and save to file.
    
    Args:
        training_stats: Dictionary containing training statistics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Training Progress (Memory Optimized)', fontsize=16)
    
    # Episode rewards
    if training_stats['episode_rewards']:
        axes[0, 0].plot(training_stats['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
    
    # Policy loss
    if training_stats['policy_losses']:
        axes[0, 1].plot(training_stats['policy_losses'])
        axes[0, 1].set_title('Policy Loss')
        axes[0, 1].set_xlabel('Update Step')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].grid(True)
    
    # Value loss
    if training_stats['value_losses']:
        axes[1, 0].plot(training_stats['value_losses'])
        axes[1, 0].set_title('Value Loss')
        axes[1, 0].set_xlabel('Update Step')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Total loss
    if training_stats['total_losses']:
        axes[1, 1].plot(training_stats['total_losses'])
        axes[1, 1].set_title('Total Loss')
        axes[1, 1].set_xlabel('Update Step')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_agent(agent: MemoryOptimizedPPOAgent, env: ConstellationEnv, num_episodes: int = 5) -> Dict[str, float]:
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained PPO agent
        env: Environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary containing evaluation statistics
    """
    episode_rewards = []
    final_ospa_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        if 'ospa_distance' in info:
            final_ospa_scores.append(info['ospa_distance'])
    
    return {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_final_ospa': np.mean(final_ospa_scores) if final_ospa_scores else None,
        'std_final_ospa': np.std(final_ospa_scores) if final_ospa_scores else None
    }


def main():
    """Main training function with memory optimization."""
    print("=" * 60)
    print("LiDAR Constellation Sensor Tasking - Memory Optimized Training")
    print("=" * 60)
    
    # Set random seeds for reproducibility
    set_random_seeds(config.SEED)
    
    # Print configuration
    print(f"Device: {config.DEVICE}")
    print(f"Number of satellites: {config.NUM_SATELLITES}")
    print(f"Simulation time steps: {config.SIMULATION_TIME_STEPS}")
    print(f"Total episodes: {config.TOTAL_EPISODES}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Memory buffer size: {getattr(config, 'PPO_MEMORY_SIZE', 5000)}")
    print()
    
    # Initialize environment, agent, and performance monitor
    print("Initializing environment, agent, and performance monitor...")
    env = ConstellationEnv()
    agent = MemoryOptimizedPPOAgent(device=config.DEVICE)
    performance_monitor = PerformanceMonitor()
    
    # Print model information
    num_params = agent.get_parameter_count()
    print(f"Model parameters: {num_params:,}")
    print()
    
    # Training loop
    print("Starting training...")
    print("-" * 60)
    
    best_reward = float('-inf')
    training_start_time = time.time()
    
    for episode in range(config.TOTAL_EPISODES):
        episode_start_time = time.time()
        
        # Monitor memory usage
        memory_usage = get_memory_usage()
        
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        # Episode loop
        while not done:
            # Select action
            action, log_prob, value = agent.select_action(state)
            
            # Take step in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, log_prob, value, done)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Update agent (PPO update)
        update_stats = agent.update()
        
        # Log performance metrics
        episode_stats = env.get_episode_statistics()
        filter_stats = env.get_filter_statistics()
        
        performance_monitor.log_episode_metrics(episode + 1, episode_stats, filter_stats)
        if update_stats:
            performance_monitor.log_training_metrics(update_stats)
        
        # Calculate episode time
        episode_time = time.time() - episode_start_time
        
        # Get latest metrics for display
        latest_metrics = performance_monitor.get_latest_metrics()
        
        # Print progress with memory information
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:4d}/{config.TOTAL_EPISODES} | "
                  f"Reward: {episode_reward:8.2f} | "
                  f"Time: {episode_time:5.2f}s | "
                  f"OSPA: {info.get('ospa_distance', 'N/A'):8.2f} | "
                  f"Tracks: {filter_stats['num_tracks']:2d} | "
                  f"Memory: {memory_usage['rss_mb']:.1f}MB")
        
        # Save model periodically
        if (episode + 1) % config.SAVE_MODEL_EVERY == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"model_episode_{episode + 1}.pth")
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")
        
        # Evaluate agent periodically
        if (episode + 1) % config.EVALUATION_EVERY == 0:
            print(f"\nEvaluating agent at episode {episode + 1}...")
            eval_stats = evaluate_agent(agent, env, num_episodes=3)  # Reduced evaluation episodes
            
            print(f"Evaluation Results:")
            print(f"  Mean episode reward: {eval_stats['mean_episode_reward']:.2f} ± {eval_stats['std_episode_reward']:.2f}")
            if eval_stats['mean_final_ospa'] is not None:
                print(f"  Mean final OSPA: {eval_stats['mean_final_ospa']:.2f} ± {eval_stats['std_final_ospa']:.2f}")
        
        # Memory cleanup
        if (episode + 1) % 25 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Memory cleanup completed. Current usage: {get_memory_usage()['rss_mb']:.1f}MB")
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print(f"Average time per episode: {total_training_time / config.TOTAL_EPISODES:.2f} seconds")
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Generate performance plots
    print("\nGenerating performance plots...")
    training_stats = agent.get_training_stats()
    plot_path = os.path.join(config.RESULTS_PLOT_PATH, "training_curves_memory_optimized.png")
    plot_training_curves(training_stats, plot_path)
    print(f"Training curves saved to {plot_path}")
    
    # Generate comprehensive performance analysis
    print("\nGenerating comprehensive performance analysis...")
    performance_monitor.create_comprehensive_plots()
    performance_monitor.create_training_curves()
    performance_monitor.create_filter_analysis()
    performance_monitor.save_metrics_to_json()
    performance_monitor.generate_performance_report()
    
    print("\n" + "=" * 60)
    print("Memory-Optimized Training Complete!")
    print("=" * 60)
    print(f"Final memory usage: {get_memory_usage()['rss_mb']:.1f}MB")
    print(f"Results saved in: {config.RESULTS_PLOT_PATH}")
    print(f"Models saved in: {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main() 