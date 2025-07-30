"""
Main Training Script for LiDAR Constellation Sensor Tasking

This script orchestrates the complete training process for the GNN-based
sensor tasking agent using PPO reinforcement learning.
"""

import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import torch
import random

# Add the simulation directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from environment.constellation_env import ConstellationEnv
from rl_agent.ppo_agent import PPOAgent
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


def plot_training_curves(training_stats: Dict[str, List[float]], save_path: str):
    """
    Plot training curves and save to file.
    
    Args:
        training_stats: Dictionary containing training statistics
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Progress', fontsize=16)
    
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


def evaluate_agent(agent: PPOAgent, env: ConstellationEnv, num_episodes: int = 10) -> Dict[str, float]:
    """
    Evaluate the trained agent.
    
    Args:
        agent: Trained PPO agent
        env: Environment
        num_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary containing evaluation metrics
    """
    agent.model.eval()
    
    episode_rewards = []
    final_ospa_scores = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            # Select action
            action, _, _ = agent.select_action(state)
            
            # Take step in environment
            state, reward, done, info = env.step(action)
            episode_reward += reward
        
        # Get episode statistics
        episode_stats = env.get_episode_statistics()
        episode_rewards.append(episode_stats['episode_reward'])
        
        if episode_stats['final_ospa'] is not None:
            final_ospa_scores.append(episode_stats['final_ospa'])
    
    return {
        'mean_episode_reward': np.mean(episode_rewards),
        'std_episode_reward': np.std(episode_rewards),
        'mean_final_ospa': np.mean(final_ospa_scores) if final_ospa_scores else None,
        'std_final_ospa': np.std(final_ospa_scores) if final_ospa_scores else None
    }


def main():
    """Main training function."""
    print("=" * 60)
    print("LiDAR Constellation Sensor Tasking - Training Script")
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
    print()
    
    # Initialize environment, agent, and performance monitor
    print("Initializing environment, agent, and performance monitor...")
    env = ConstellationEnv()
    agent = PPOAgent(device=config.DEVICE)
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
    
    # Performance optimization: disable gradient computation for evaluation
    torch.set_grad_enabled(True)
    
    for episode in range(config.TOTAL_EPISODES):
        episode_start_time = time.time()
        
        # Reset environment
        state = env.reset()
        episode_reward = 0.0
        done = False
        
        # Episode loop
        timestep_count = 0
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
            timestep_count += 1
            
            # Debug: Print timestep progress every 25 timesteps (full episode)
            # Removed redundant print statement - now showing detailed progress for every episode
        
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
        
        # Print progress (every episode)
        print(f"Episode {episode + 1:4d}/{config.TOTAL_EPISODES} | "
              f"Reward: {episode_reward:8.2f} | "
              f"Time: {episode_time:5.2f}s | "
              f"OSPA: {info.get('ospa_distance', 'N/A'):8.2f} | "
              f"Tracks: {filter_stats['num_tracks']:2d} | "
              f"Det. Rate: {filter_stats['detection_rate']:.3f}")
        
        # Save model periodically
        if (episode + 1) % config.SAVE_MODEL_EVERY == 0:
            model_path = os.path.join(config.MODEL_SAVE_PATH, f"model_episode_{episode + 1}.pth")
            agent.save_model(model_path)
            print(f"Model saved to {model_path}")
        
        # Evaluate agent periodically
        if (episode + 1) % config.EVALUATION_EVERY == 0:
            print(f"\nEvaluating agent at episode {episode + 1}...")
            eval_stats = evaluate_agent(agent, env, num_episodes=5)
            
            print(f"Evaluation Results:")
            print(f"  Mean episode reward: {eval_stats['mean_episode_reward']:.2f} ± {eval_stats['std_episode_reward']:.2f}")
            if eval_stats['mean_final_ospa'] is not None:
                print(f"  Mean final OSPA: {eval_stats['mean_final_ospa']:.2f} ± {eval_stats['std_final_ospa']:.2f}")
            print()
            
            # Save best model
            if eval_stats['mean_episode_reward'] > best_reward:
                best_reward = eval_stats['mean_episode_reward']
                best_model_path = os.path.join(config.MODEL_SAVE_PATH, "best_model.pth")
                agent.save_model(best_model_path)
                print(f"New best model saved! (Reward: {best_reward:.2f})")
                print()
        
        # Create performance plots periodically
        if (episode + 1) % 100 == 0:
            # Create comprehensive plots
            performance_monitor.create_comprehensive_plots()
            performance_monitor.create_training_curves()
            performance_monitor.create_filter_analysis()
            
            # Save metrics to JSON
            performance_monitor.save_metrics_to_json()
    
    # Training completed
    total_training_time = time.time() - training_start_time
    print("=" * 60)
    print("Training completed!")
    print(f"Total training time: {total_training_time / 3600:.2f} hours")
    print()
    
    # Final evaluation
    print("Performing final evaluation...")
    final_eval_stats = evaluate_agent(agent, env, num_episodes=20)
    
    print("Final Evaluation Results:")
    print(f"  Mean episode reward: {final_eval_stats['mean_episode_reward']:.2f} ± {final_eval_stats['std_episode_reward']:.2f}")
    if final_eval_stats['mean_final_ospa'] is not None:
        print(f"  Mean final OSPA: {final_eval_stats['mean_final_ospa']:.2f} ± {final_eval_stats['std_final_ospa']:.2f}")
    print()
    
    # Save final model
    final_model_path = os.path.join(config.MODEL_SAVE_PATH, "final_model.pth")
    agent.save_model(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Create final performance analysis
    print("\nGenerating final performance analysis...")
    performance_monitor.create_comprehensive_plots()
    performance_monitor.create_training_curves()
    performance_monitor.create_filter_analysis()
    performance_monitor.save_metrics_to_json()
    performance_monitor.generate_performance_report()
    
    print("Final performance analysis completed!")
    
    print("=" * 60)
    print("Training script completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main() 