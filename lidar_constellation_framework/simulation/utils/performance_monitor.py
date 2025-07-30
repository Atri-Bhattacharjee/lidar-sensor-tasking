"""
Performance Monitoring and Visualization System

This module provides comprehensive tracking and visualization of performance metrics
for both the reinforcement learning agent and the LMB filter.
"""

import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
import config


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for the LiDAR constellation framework.
    """
    
    def __init__(self, save_dir: str = None):
        """
        Initialize the performance monitor.
        
        Args:
            save_dir: Directory to save performance data and plots
        """
        self.save_dir = save_dir or config.RESULTS_PLOT_PATH
        os.makedirs(self.save_dir, exist_ok=True)
        
        # Initialize metric storage
        self.metrics = defaultdict(list)
        self.episode_metrics = defaultdict(list)
        self.filter_metrics = defaultdict(list)
        self.training_stats = defaultdict(list)
        
        # Real-time plotting
        self.live_plot = False
        self.plot_interval = 10  # Update plots every N episodes
        
        # Performance tracking
        self.start_time = time.time()
        self.episode_times = deque(maxlen=100)
        
        # Initialize plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def log_episode_metrics(self, episode: int, episode_stats: Dict[str, Any], 
                           filter_stats: Dict[str, Any] = None):
        """
        Log metrics for a single episode.
        
        Args:
            episode: Episode number
            episode_stats: Episode statistics from environment
            filter_stats: Filter performance statistics
        """
        # Episode metrics
        self.episode_metrics['episode'].append(episode)
        self.episode_metrics['reward'].append(episode_stats.get('episode_reward', 0.0))
        self.episode_metrics['final_ospa'].append(episode_stats.get('final_ospa', None))
        self.episode_metrics['mean_ospa'].append(episode_stats.get('mean_ospa', None))
        self.episode_metrics['num_timesteps'].append(episode_stats.get('num_timesteps', 0))
        
        # Filter metrics (if available)
        if filter_stats:
            self.filter_metrics['episode'].append(episode)
            self.filter_metrics['num_tracks'].append(filter_stats.get('num_tracks', 0))
            self.filter_metrics['num_measurements'].append(filter_stats.get('num_measurements', 0))
            self.filter_metrics['detection_rate'].append(filter_stats.get('detection_rate', 0.0))
            self.filter_metrics['false_alarm_rate'].append(filter_stats.get('false_alarm_rate', 0.0))
            self.filter_metrics['track_accuracy'].append(filter_stats.get('track_accuracy', 0.0))
        
        # Timing
        episode_time = time.time() - self.start_time
        self.episode_times.append(episode_time)
        self.episode_metrics['episode_time'].append(episode_time)
        
        # Update live plots if enabled
        if self.live_plot and episode % self.plot_interval == 0:
            self.update_live_plots()
    
    def log_training_metrics(self, update_stats: Dict[str, float]):
        """
        Log training metrics from PPO updates.
        
        Args:
            update_stats: Training statistics from PPO agent
        """
        for key, value in update_stats.items():
            self.training_stats[key].append(value)
    
    def log_filter_performance(self, filter_stats: Dict[str, Any]):
        """
        Log detailed filter performance metrics.
        
        Args:
            filter_stats: Filter performance statistics
        """
        for key, value in filter_stats.items():
            self.filter_metrics[key].append(value)
    
    def create_comprehensive_plots(self, save_path: str = None):
        """
        Create comprehensive performance plots.
        
        Args:
            save_path: Path to save the plots
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"performance_analysis_{timestamp}.png")
        
        # Create subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Training Progress (2x2 grid)
        gs1 = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Episode rewards
        ax1 = fig.add_subplot(gs1[0, :2])
        if self.episode_metrics['reward']:
            ax1.plot(self.episode_metrics['episode'], self.episode_metrics['reward'], 
                    alpha=0.7, linewidth=1)
            ax1.set_title('Episode Rewards', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Reward')
            ax1.grid(True, alpha=0.3)
        
        # OSPA distance
        ax2 = fig.add_subplot(gs1[0, 2:])
        if self.episode_metrics['final_ospa']:
            valid_ospa = [(ep, ospa) for ep, ospa in zip(self.episode_metrics['episode'], 
                                                        self.episode_metrics['final_ospa']) 
                         if ospa is not None]
            if valid_ospa:
                episodes, ospa_values = zip(*valid_ospa)
                ax2.plot(episodes, ospa_values, alpha=0.7, linewidth=1, color='red')
                ax2.set_title('OSPA Distance', fontsize=12, fontweight='bold')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('OSPA Distance')
                ax2.grid(True, alpha=0.3)
        
        # Training losses
        ax3 = fig.add_subplot(gs1[1, :2])
        if self.training_stats['policy_losses']:
            ax3.plot(self.training_stats['policy_losses'], label='Policy Loss', alpha=0.7)
            ax3.plot(self.training_stats['value_losses'], label='Value Loss', alpha=0.7)
            ax3.plot(self.training_stats['total_losses'], label='Total Loss', alpha=0.7)
            ax3.set_title('Training Losses', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Update Step')
            ax3.set_ylabel('Loss')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Entropy
        ax4 = fig.add_subplot(gs1[1, 2:])
        if self.training_stats['entropy_losses']:
            ax4.plot(self.training_stats['entropy_losses'], color='green', alpha=0.7)
            ax4.set_title('Policy Entropy', fontsize=12, fontweight='bold')
            ax4.set_xlabel('Update Step')
            ax4.set_ylabel('Entropy')
            ax4.grid(True, alpha=0.3)
        
        # 2. Filter Performance (2x2 grid)
        # Number of tracks
        ax5 = fig.add_subplot(gs1[2, :2])
        if self.filter_metrics['num_tracks']:
            ax5.plot(self.filter_metrics['episode'], self.filter_metrics['num_tracks'], 
                    alpha=0.7, linewidth=1, color='purple')
            ax5.set_title('Number of Tracks', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Episode')
            ax5.set_ylabel('Number of Tracks')
            ax5.grid(True, alpha=0.3)
        
        # Detection rate
        ax6 = fig.add_subplot(gs1[2, 2:])
        if self.filter_metrics['detection_rate']:
            ax6.plot(self.filter_metrics['episode'], self.filter_metrics['detection_rate'], 
                    alpha=0.7, linewidth=1, color='orange')
            ax6.set_title('Detection Rate', fontsize=12, fontweight='bold')
            ax6.set_xlabel('Episode')
            ax6.set_ylabel('Detection Rate')
            ax6.grid(True, alpha=0.3)
        
        # Track accuracy
        ax7 = fig.add_subplot(gs1[3, :2])
        if self.filter_metrics['track_accuracy']:
            ax7.plot(self.filter_metrics['episode'], self.filter_metrics['track_accuracy'], 
                    alpha=0.7, linewidth=1, color='brown')
            ax7.set_title('Track Accuracy', fontsize=12, fontweight='bold')
            ax7.set_xlabel('Episode')
            ax7.set_ylabel('Accuracy')
            ax7.grid(True, alpha=0.3)
        
        # False alarm rate
        ax8 = fig.add_subplot(gs1[3, 2:])
        if self.filter_metrics['false_alarm_rate']:
            ax8.plot(self.filter_metrics['episode'], self.filter_metrics['false_alarm_rate'], 
                    alpha=0.7, linewidth=1, color='pink')
            ax8.set_title('False Alarm Rate', fontsize=12, fontweight='bold')
            ax8.set_xlabel('Episode')
            ax8.set_ylabel('False Alarm Rate')
            ax8.grid(True, alpha=0.3)
        
        plt.suptitle('LiDAR Constellation Performance Analysis', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comprehensive performance plots saved to: {save_path}")
    
    def create_training_curves(self, save_path: str = None):
        """
        Create focused training curves plot.
        
        Args:
            save_path: Path to save the plot
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"training_curves_{timestamp}.png")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Episode rewards
        if self.episode_metrics['reward']:
            axes[0, 0].plot(self.episode_metrics['episode'], self.episode_metrics['reward'], 
                           alpha=0.7, linewidth=1)
            axes[0, 0].set_title('Episode Rewards', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Reward')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Policy loss
        if self.training_stats['policy_losses']:
            axes[0, 1].plot(self.training_stats['policy_losses'], alpha=0.7)
            axes[0, 1].set_title('Policy Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Update Step')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Value loss
        if self.training_stats['value_losses']:
            axes[1, 0].plot(self.training_stats['value_losses'], alpha=0.7, color='orange')
            axes[1, 0].set_title('Value Loss', fontweight='bold')
            axes[1, 0].set_xlabel('Update Step')
            axes[1, 0].set_ylabel('Loss')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Total loss
        if self.training_stats['total_losses']:
            axes[1, 1].plot(self.training_stats['total_losses'], alpha=0.7, color='red')
            axes[1, 1].set_title('Total Loss', fontweight='bold')
            axes[1, 1].set_xlabel('Update Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Training curves saved to: {save_path}")
    
    def create_filter_analysis(self, save_path: str = None):
        """
        Create detailed filter performance analysis.
        
        Args:
            save_path: Path to save the plot
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"filter_analysis_{timestamp}.png")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('LMB Filter Performance Analysis', fontsize=16, fontweight='bold')
        
        # Number of tracks over time
        if self.filter_metrics['num_tracks']:
            axes[0, 0].plot(self.filter_metrics['episode'], self.filter_metrics['num_tracks'], 
                           alpha=0.7, linewidth=1)
            axes[0, 0].set_title('Number of Tracks', fontweight='bold')
            axes[0, 0].set_xlabel('Episode')
            axes[0, 0].set_ylabel('Number of Tracks')
            axes[0, 0].grid(True, alpha=0.3)
        
        # Number of measurements
        if self.filter_metrics['num_measurements']:
            axes[0, 1].plot(self.filter_metrics['episode'], self.filter_metrics['num_measurements'], 
                           alpha=0.7, linewidth=1, color='green')
            axes[0, 1].set_title('Number of Measurements', fontweight='bold')
            axes[0, 1].set_xlabel('Episode')
            axes[0, 1].set_ylabel('Number of Measurements')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Detection rate
        if self.filter_metrics['detection_rate']:
            axes[0, 2].plot(self.filter_metrics['episode'], self.filter_metrics['detection_rate'], 
                           alpha=0.7, linewidth=1, color='blue')
            axes[0, 2].set_title('Detection Rate', fontweight='bold')
            axes[0, 2].set_xlabel('Episode')
            axes[0, 2].set_ylabel('Detection Rate')
            axes[0, 2].grid(True, alpha=0.3)
        
        # False alarm rate
        if self.filter_metrics['false_alarm_rate']:
            axes[1, 0].plot(self.filter_metrics['episode'], self.filter_metrics['false_alarm_rate'], 
                           alpha=0.7, linewidth=1, color='red')
            axes[1, 0].set_title('False Alarm Rate', fontweight='bold')
            axes[1, 0].set_xlabel('Episode')
            axes[1, 0].set_ylabel('False Alarm Rate')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Track accuracy
        if self.filter_metrics['track_accuracy']:
            axes[1, 1].plot(self.filter_metrics['episode'], self.filter_metrics['track_accuracy'], 
                           alpha=0.7, linewidth=1, color='purple')
            axes[1, 1].set_title('Track Accuracy', fontweight='bold')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Combined metrics
        if self.filter_metrics['detection_rate'] and self.filter_metrics['false_alarm_rate']:
            ax = axes[1, 2]
            ax.plot(self.filter_metrics['episode'], self.filter_metrics['detection_rate'], 
                   label='Detection Rate', alpha=0.7)
            ax.plot(self.filter_metrics['episode'], self.filter_metrics['false_alarm_rate'], 
                   label='False Alarm Rate', alpha=0.7)
            ax.set_title('Detection vs False Alarm Rate', fontweight='bold')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Rate')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Filter analysis saved to: {save_path}")
    
    def update_live_plots(self):
        """Update live plots during training."""
        # This could be implemented with matplotlib's interactive mode
        # For now, we'll just save plots periodically
        pass
    
    def save_metrics_to_json(self, save_path: str = None):
        """
        Save all metrics to a JSON file for later analysis.
        
        Args:
            save_path: Path to save the JSON file
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"metrics_{timestamp}.json")
        
        # Convert numpy arrays to lists for JSON serialization
        metrics_dict = {}
        for key, values in self.episode_metrics.items():
            metrics_dict[f'episode_{key}'] = [float(v) if v is not None else None for v in values]
        
        for key, values in self.training_stats.items():
            metrics_dict[f'training_{key}'] = [float(v) for v in values]
        
        for key, values in self.filter_metrics.items():
            metrics_dict[f'filter_{key}'] = [float(v) if v is not None else None for v in values]
        
        # Add metadata
        metrics_dict['metadata'] = {
            'total_episodes': len(self.episode_metrics['episode']),
            'total_training_updates': len(self.training_stats['policy_losses']),
            'start_time': self.start_time,
            'end_time': time.time(),
            'config': {
                'num_satellites': config.NUM_SATELLITES,
                'simulation_time_steps': config.SIMULATION_TIME_STEPS,
                'learning_rate': config.LEARNING_RATE,
                'batch_size': config.BATCH_SIZE
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        print(f"Metrics saved to: {save_path}")
    
    def generate_performance_report(self, save_path: str = None):
        """
        Generate a comprehensive performance report.
        
        Args:
            save_path: Path to save the report
        """
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(self.save_dir, f"performance_report_{timestamp}.txt")
        
        with open(save_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("LiDAR Constellation Performance Report\n")
            f.write("=" * 60 + "\n\n")
            
            # Training statistics
            f.write("TRAINING STATISTICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Episodes: {len(self.episode_metrics['episode'])}\n")
            f.write(f"Total Training Updates: {len(self.training_stats['policy_losses'])}\n")
            f.write(f"Total Training Time: {time.time() - self.start_time:.2f} seconds\n")
            
            if self.episode_metrics['reward']:
                f.write(f"Average Episode Reward: {np.mean(self.episode_metrics['reward']):.4f}\n")
                f.write(f"Best Episode Reward: {np.max(self.episode_metrics['reward']):.4f}\n")
                f.write(f"Worst Episode Reward: {np.min(self.episode_metrics['reward']):.4f}\n")
            
            if self.episode_metrics['final_ospa']:
                valid_ospa = [ospa for ospa in self.episode_metrics['final_ospa'] if ospa is not None]
                if valid_ospa:
                    f.write(f"Average Final OSPA: {np.mean(valid_ospa):.4f}\n")
                    f.write(f"Best Final OSPA: {np.min(valid_ospa):.4f}\n")
                    f.write(f"Worst Final OSPA: {np.max(valid_ospa):.4f}\n")
            
            # Filter statistics
            if self.filter_metrics['num_tracks']:
                f.write("\nFILTER STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Average Number of Tracks: {np.mean(self.filter_metrics['num_tracks']):.2f}\n")
                f.write(f"Average Detection Rate: {np.mean(self.filter_metrics['detection_rate']):.4f}\n")
                f.write(f"Average False Alarm Rate: {np.mean(self.filter_metrics['false_alarm_rate']):.4f}\n")
                f.write(f"Average Track Accuracy: {np.mean(self.filter_metrics['track_accuracy']):.4f}\n")
            
            # Training loss statistics
            if self.training_stats['policy_losses']:
                f.write("\nTRAINING LOSS STATISTICS:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Final Policy Loss: {self.training_stats['policy_losses'][-1]:.6f}\n")
                f.write(f"Final Value Loss: {self.training_stats['value_losses'][-1]:.6f}\n")
                f.write(f"Final Total Loss: {self.training_stats['total_losses'][-1]:.6f}\n")
                f.write(f"Average Policy Loss: {np.mean(self.training_stats['policy_losses']):.6f}\n")
                f.write(f"Average Value Loss: {np.mean(self.training_stats['value_losses']):.6f}\n")
        
        print(f"Performance report saved to: {save_path}")
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """
        Get the latest metrics for monitoring.
        
        Returns:
            Dictionary containing the latest metrics
        """
        latest = {}
        
        if self.episode_metrics['reward']:
            latest['latest_reward'] = self.episode_metrics['reward'][-1]
            latest['avg_reward'] = np.mean(self.episode_metrics['reward'][-10:])  # Last 10 episodes
        
        if self.episode_metrics['final_ospa']:
            valid_ospa = [ospa for ospa in self.episode_metrics['final_ospa'][-10:] if ospa is not None]
            if valid_ospa:
                latest['latest_ospa'] = valid_ospa[-1]
                latest['avg_ospa'] = np.mean(valid_ospa)
        
        if self.training_stats['total_losses']:
            latest['latest_loss'] = self.training_stats['total_losses'][-1]
            latest['avg_loss'] = np.mean(self.training_stats['total_losses'][-10:])
        
        return latest 