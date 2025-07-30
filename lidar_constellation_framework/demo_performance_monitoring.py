"""
Demonstration of Performance Monitoring System

This script demonstrates how to use the performance monitoring system
to track and visualize training progress and filter performance.
"""

import sys
import os
import numpy as np
import time

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from utils.performance_monitor import PerformanceMonitor
import config


def demo_performance_monitoring():
    """Demonstrate the performance monitoring system."""
    print("=" * 60)
    print("Performance Monitoring System Demo")
    print("=" * 60)
    
    # Initialize performance monitor
    monitor = PerformanceMonitor()
    
    # Simulate training progress
    print("\nSimulating training progress...")
    
    for episode in range(50):
        # Simulate episode statistics
        episode_stats = {
            'episode_reward': np.random.normal(-100, 50),  # Simulated reward
            'final_ospa': np.random.uniform(50, 200),      # Simulated OSPA
            'mean_ospa': np.random.uniform(40, 180),       # Simulated mean OSPA
            'num_timesteps': config.SIMULATION_TIME_STEPS
        }
        
        # Simulate filter statistics
        filter_stats = {
            'num_tracks': np.random.randint(10, 30),
            'num_measurements': np.random.randint(50, 150),
            'detection_rate': np.random.uniform(0.6, 0.95),
            'false_alarm_rate': np.random.uniform(0.05, 0.3),
            'track_accuracy': np.random.uniform(0.7, 0.95)
        }
        
        # Simulate training statistics (every 5 episodes)
        if episode % 5 == 0:
            training_stats = {
                'policy_losses': np.random.uniform(0.1, 0.5),
                'value_losses': np.random.uniform(0.05, 0.3),
                'entropy_losses': np.random.uniform(0.01, 0.1),
                'total_losses': np.random.uniform(0.2, 0.8)
            }
            monitor.log_training_metrics(training_stats)
        
        # Log episode metrics
        monitor.log_episode_metrics(episode + 1, episode_stats, filter_stats)
        
        # Print progress
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1:2d} | "
                  f"Reward: {episode_stats['episode_reward']:6.1f} | "
                  f"OSPA: {episode_stats['final_ospa']:5.1f} | "
                  f"Tracks: {filter_stats['num_tracks']:2d} | "
                  f"Det. Rate: {filter_stats['detection_rate']:.3f}")
        
        # Small delay to simulate training time
        time.sleep(0.01)
    
    print("\nGenerating performance analysis...")
    
    # Create all types of plots
    monitor.create_comprehensive_plots()
    monitor.create_training_curves()
    monitor.create_filter_analysis()
    
    # Save metrics and generate report
    monitor.save_metrics_to_json()
    monitor.generate_performance_report()
    
    print("\nDemo completed! Check the results/plots directory for generated files.")
    print("\nGenerated files:")
    print("- performance_analysis_*.png (comprehensive analysis)")
    print("- training_curves_*.png (training progress)")
    print("- filter_analysis_*.png (filter performance)")
    print("- metrics_*.json (raw data)")
    print("- performance_report_*.txt (text report)")


def demo_live_monitoring():
    """Demonstrate live monitoring capabilities."""
    print("\n" + "=" * 60)
    print("Live Monitoring Demo")
    print("=" * 60)
    
    monitor = PerformanceMonitor()
    
    print("Simulating live training with periodic monitoring...")
    
    for episode in range(20):
        # Simulate metrics
        episode_stats = {
            'episode_reward': np.random.normal(-80, 40),
            'final_ospa': np.random.uniform(60, 150),
            'mean_ospa': np.random.uniform(50, 130),
            'num_timesteps': config.SIMULATION_TIME_STEPS
        }
        
        filter_stats = {
            'num_tracks': np.random.randint(15, 25),
            'num_measurements': np.random.randint(80, 120),
            'detection_rate': np.random.uniform(0.7, 0.9),
            'false_alarm_rate': np.random.uniform(0.1, 0.25),
            'track_accuracy': np.random.uniform(0.75, 0.9)
        }
        
        if episode % 3 == 0:
            training_stats = {
                'policy_losses': np.random.uniform(0.15, 0.4),
                'value_losses': np.random.uniform(0.08, 0.25),
                'entropy_losses': np.random.uniform(0.02, 0.08),
                'total_losses': np.random.uniform(0.25, 0.7)
            }
            monitor.log_training_metrics(training_stats)
        
        monitor.log_episode_metrics(episode + 1, episode_stats, filter_stats)
        
        # Get latest metrics for monitoring
        latest = monitor.get_latest_metrics()
        
        print(f"Episode {episode + 1:2d} | "
              f"Latest Reward: {latest.get('latest_reward', 0):6.1f} | "
              f"Avg Reward: {latest.get('avg_reward', 0):6.1f} | "
              f"Latest OSPA: {latest.get('latest_ospa', 0):5.1f} | "
              f"Avg OSPA: {latest.get('avg_ospa', 0):5.1f}")
        
        time.sleep(0.02)
    
    print("\nLive monitoring demo completed!")


if __name__ == "__main__":
    # Run demos
    demo_performance_monitoring()
    demo_live_monitoring()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60) 