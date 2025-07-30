"""
Constellation Environment for LiDAR Sensor Tasking

This module implements the main reinforcement learning environment that orchestrates
the entire simulation loop, managing the state, actions, and rewards.
"""

import gym
import numpy as np
import torch
import pickle
import os
from typing import Dict, List, Tuple, Any
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment
import config

from .perception_layer import PerceptionLayer
from .estimation_layer import EstimationLayer
from utils.ospa import calculate_ospa


class ConstellationEnv(gym.Env):
    """
    Reinforcement learning environment for LiDAR constellation sensor tasking.
    
    This environment manages the complete simulation loop including:
    - Sensor tasking decisions
    - Measurement generation
    - State estimation
    - Reward calculation
    """
    
    def __init__(self):
        """Initialize the constellation environment."""
        super(ConstellationEnv, self).__init__()
        
        # Initialize perception and estimation layers
        self.perception_layer = PerceptionLayer()
        self.estimation_layer = EstimationLayer(perception_layer=self.perception_layer)
        
        # Load the pre-processed ground truth database
        ground_truth_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ground_truth_data", "synthetic_ground_truth_database.pkl")
        
        if os.path.exists(ground_truth_path):
            print(f"Loading synthetic ground truth database from: {ground_truth_path}")
            with open(ground_truth_path, 'rb') as f:
                self.ground_truth = pickle.load(f)
            print(f"Loaded synthetic ground truth database with {len(self.ground_truth)} timesteps")
            print(f"First timestep contains {len(self.ground_truth[0])} debris objects")
        else:
            print(f"Warning: Synthetic ground truth database not found at {ground_truth_path}")
            print("Falling back to old synthetic ground truth data")
            self.ground_truth = self._generate_synthetic_ground_truth()
        
        # Initialize state variables
        self.unassociated_measurements = []
        self.current_time_step = 0
        
        # Define action space: continuous azimuth and elevation for each satellite
        # Shape: (NUM_SATELLITES, 2) with values in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(config.NUM_SATELLITES, 2),
            dtype=np.float32
        )
        
        # Define observation space: graph representation of LMB filter state
        # This is a complex space represented as a dictionary
        self.observation_space = gym.spaces.Dict({
            'node_features': gym.spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(1000, 13),  # Maximum 1000 nodes, 13 features each
                dtype=np.float32
            ),
            'edge_index': gym.spaces.Box(
                low=0,
                high=1000,
                shape=(2, 5000),  # Maximum 5000 edges
                dtype=np.int64
            ),
            'num_nodes': gym.spaces.Discrete(1000)  # Maximum number of nodes
        })
        
        # Episode statistics
        self.episode_reward = 0.0
        self.episode_ospa_scores = []
    
    def _generate_synthetic_ground_truth(self) -> Dict[int, List[Dict]]:
        """
        Generate synthetic ground truth data for the simulation.
        
        Returns:
            Dictionary mapping timesteps to lists of ground truth objects
        """
        # This method is deprecated - ground truth should be loaded from processed CPE files
        # In a real implementation, this would load from the ground truth database
        
        print("Warning: Using synthetic ground truth data. This should be replaced with actual data from CPE files.")
        
        ground_truth_by_timestep = {}
        
        # Generate 50 debris objects with realistic orbital parameters
        num_objects = 50
        
        for timestep in range(config.SIMULATION_TIME_STEPS):
            timestep_objects = []
            
            for obj_id in range(num_objects):
                # Generate orbital parameters similar to the actual constellation
                # Use realistic LEO parameters based on the .dia files
                altitude = 650000.0 + np.random.uniform(-50000, 50000)  # 600-700 km altitude
                inclination = 98.6 + np.random.uniform(-5, 5)  # Near SSO inclination
                raan = np.random.uniform(0, 360)  # Random RAAN
                mean_anomaly = np.random.uniform(0, 360)  # Random position in orbit
                
                # Convert to Cartesian coordinates
                radius = config.EARTH_RADIUS_M + altitude
                
                # Calculate position using proper orbital mechanics
                x = radius * np.cos(np.radians(mean_anomaly)) * np.cos(np.radians(raan))
                y = radius * np.sin(np.radians(mean_anomaly)) * np.cos(np.radians(inclination))
                z = radius * np.sin(np.radians(mean_anomaly)) * np.sin(np.radians(inclination))
                
                # Calculate velocity (circular orbit approximation)
                orbital_velocity = np.sqrt(config.GRAVITATIONAL_CONSTANT / radius)
                vx = -orbital_velocity * np.sin(np.radians(mean_anomaly)) * np.cos(np.radians(raan))
                vy = orbital_velocity * np.cos(np.radians(mean_anomaly)) * np.cos(np.radians(inclination))
                vz = orbital_velocity * np.cos(np.radians(mean_anomaly)) * np.sin(np.radians(inclination))
                
                # Create object data
                object_data = {
                    'UniqueID': f"SYNTH_{obj_id}",
                    'diameter': np.random.uniform(0.01, 1.0),  # 1cm to 1m
                    'mass': np.random.uniform(0.1, 100.0),  # 0.1 to 100 kg
                    'position': np.array([x, y, z]),
                    'velocity': np.array([vx, vy, vz]),
                    'state_vector': np.array([x, y, z, vx, vy, vz])
                }
                
                timestep_objects.append(object_data)
            
            ground_truth_by_timestep[timestep] = timestep_objects
        
        return ground_truth_by_timestep
    
    def _build_graph_from_state(self, extracted_state: List[Dict]) -> Data:
        """
        Convert the LMB filter's extracted state into a graph for the GNN.
        
        Args:
            extracted_state: List of tracked objects from the estimation layer
            
        Returns:
            PyTorch Geometric Data object representing the graph
        """
        if not extracted_state:
            # Empty state: create graph with one dummy node and self-loop to avoid GAT issues
            return Data(
                x=torch.zeros((1, 13), dtype=torch.float32),
                edge_index=torch.tensor([[0, 0]], dtype=torch.long).t(),  # Self-loop
                batch=torch.zeros(1, dtype=torch.long)
            )
        
        # Extract node features
        node_features = []
        for obj in extracted_state:
            # Concatenate: [6D state + 6D covariance diagonal + 1D existence probability]
            state_vector = np.array(obj['mean'])
            covariance_diag = np.diag(obj['covariance'])
            existence_prob = obj['existence_probability']
            
            # Combine into 13-dimensional feature vector
            feature_vector = np.concatenate([
                state_vector,      # 6D state [Px, Py, Pz, Vx, Vy, Vz]
                covariance_diag,   # 6D covariance diagonal
                [existence_prob]   # 1D existence probability
            ])
            
            node_features.append(feature_vector)
        
        node_features = np.array(node_features)
        
        # Create edge connectivity using k-Nearest Neighbors
        if len(node_features) > 1:
            # Use 3D positions for k-NN (first 3 dimensions)
            positions_3d = node_features[:, :3]
            
            # Find k-nearest neighbors
            nbrs = NearestNeighbors(
                n_neighbors=min(config.K_NEAREST_NEIGHBORS, len(node_features) - 1),
                algorithm='ball_tree'
            ).fit(positions_3d)
            
            distances, indices = nbrs.kneighbors(positions_3d)
            
            # Create edge index
            edge_list = []
            for i, neighbors in enumerate(indices):
                for j in neighbors[1:]:  # Skip self-connection
                    edge_list.append([i, j])
                    edge_list.append([j, i])  # Undirected graph
            
            # Add self-loops for all nodes
            for i in range(len(node_features)):
                edge_list.append([i, i])
            
            edge_index = np.array(edge_list).T
        else:
            # Single node: create self-loop
            edge_index = np.array([[0, 0]], dtype=int).T
        
        # Convert to PyTorch tensors
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index)
        batch = torch.zeros(len(node_features), dtype=torch.long)
        
        # Debug: Check if edge_index is empty and fix it
        if edge_index.size(1) == 0:
            # Create self-loops for all nodes
            num_nodes = len(node_features)
            self_loops = torch.arange(num_nodes, dtype=torch.long).unsqueeze(0).repeat(2, 1)
            edge_index = self_loops
        
        # Final safety check: ensure edge_index is never empty
        if edge_index.size(1) == 0:
            print(f"Warning: edge_index is still empty after fixes. Creating default self-loops.")
            num_nodes = len(node_features)
            edge_index = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t()
        
        return Data(x=x, edge_index=edge_index, batch=batch)
    
    def reset(self) -> Data:
        """
        Reset the environment for a new episode.
        
        Returns:
            Initial graph observation
        """
        # Reset time step
        self.current_time_step = 0
        
        # Reset estimation layer
        self.estimation_layer = EstimationLayer()
        
        # Reset unassociated measurements
        self.unassociated_measurements = []
        
        # Reset episode statistics
        self.episode_reward = 0.0
        self.episode_ospa_scores = []
        
        # Get initial belief state (empty)
        initial_state = self.estimation_layer._extract_state()
        
        # Build initial graph observation
        initial_observation = self._build_graph_from_state(initial_state)
        
        return initial_observation
    
    def step(self, action: np.ndarray) -> Tuple[Data, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Array of shape (NUM_SATELLITES, 2) with normalized pointing directions
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        # Increment time step
        self.current_time_step += 1
        
        # Ensure action has the correct shape
        if action.size == 0 or action.shape[0] == 0:
            # If action is empty, create a default action
            action = np.zeros((config.NUM_SATELLITES, 2), dtype=np.float32)
        elif len(action.shape) == 3 and action.shape[0] == 0:
            # If batch dimension is 0, squeeze it out
            action = action.squeeze(0)
        elif len(action.shape) == 3 and action.shape[0] == 1:
            # If batch dimension is 1, squeeze it out to get (40, 2)
            action = action.squeeze(0)
        
        # De-normalize action: scale from [-1, 1] to physical pointing limits
        # Azimuth: [-1, 1] -> [-π/2, π/2] (more reasonable range)
        # Elevation: [-1, 1] -> [-π/6, π/6] (more reasonable range)
        denormalized_action = action.copy()
        denormalized_action[:, 0] *= np.pi / 2  # Azimuth (reduced range)
        denormalized_action[:, 1] *= np.pi / 6  # Elevation (reduced range)
        
        # Clip actions to prevent impossible pointing
        denormalized_action[:, 0] = np.clip(denormalized_action[:, 0], -np.pi/2, np.pi/2)
        denormalized_action[:, 1] = np.clip(denormalized_action[:, 1], -np.pi/6, np.pi/6)
        
        # Get current ground truth: Directly access the list of objects for the current timestep
        current_objects = self.ground_truth.get(self.current_time_step, [])
        
        # Debug output
        if self.current_time_step % 10 == 0:  # Print every 10 timesteps
            display_timestep = self.current_time_step + 1  # Show 1-50 instead of 0-49
            print(f"Timestep {display_timestep}: Processing {len(current_objects)} ground truth objects")
        

        
        # Convert to the format expected by the perception layer
        ground_truth_at_t = []
        for obj in current_objects:
            # Map the synthetic debris keys to expected format
            ground_truth_at_t.append({
                'id': obj.get('debris_id', obj.get('id', obj.get('UniqueID', f"obj_{len(ground_truth_at_t)}"))),
                'position': obj.get('position', np.array([0.0, 0.0, 0.0])),
                'velocity': obj.get('velocity', np.array([0.0, 0.0, 0.0])),
                'diameter': obj.get('diameter', 1.0),
                'mass': obj.get('mass', 100.0)
            })
        
        # Debug output
        if self.current_time_step % 10 == 0:  # Print every 10 timesteps
            display_timestep = self.current_time_step + 1  # Show 1-50 instead of 0-49
            print(f"Timestep {display_timestep}: Converted {len(ground_truth_at_t)} objects for perception layer")
        
        # Calculate time delta
        dt = config.EPISODE_DURATION_SECONDS / config.SIMULATION_TIME_STEPS
        
        # Calculate current time in seconds
        current_time_seconds = self.current_time_step * dt
        
        # Generate measurements using perception layer with current time
        measurements = self.perception_layer.generate_measurements(
            ground_truth_at_t, denormalized_action, current_time_seconds
        )
        

        
        # Update estimation layer
        extracted_state, new_unassociated_measurements = self.estimation_layer.step(
            measurements, self.unassociated_measurements, dt
        )
        

        
        # Update unassociated measurements for next step
        self.unassociated_measurements = new_unassociated_measurements
        
        # Build graph observation
        observation = self._build_graph_from_state(extracted_state)
        
        # Ensure the graph has a proper batch assignment
        if not hasattr(observation, 'batch') or observation.batch is None:
            observation.batch = torch.zeros(observation.x.size(0), dtype=torch.long)
        
        # Check if episode is done
        done = self.current_time_step >= config.SIMULATION_TIME_STEPS - 1  # Stop at timestep 49, not 50
        
        # Calculate reward with intermediate feedback
        reward = 0.0
        
        # Initialize variables that will be used in info dictionary
        current_ospa = config.OSPA_CUTOFF_C  # Default to max penalty
        detection_rate = 0.0  # Default detection rate
        num_tracks = len(extracted_state)  # Number of extracted tracks
        num_ground_truth = len(ground_truth_at_t)  # Always calculate this
        
        # Calculate OSPA and detection rate every 5 timesteps or when done
        if self.current_time_step % 5 == 0 or done:
            # Calculate current OSPA distance
            current_ospa = calculate_ospa(
                ground_truth_at_t, extracted_state,
                config.OSPA_CUTOFF_C, config.OSPA_ORDER_P
            )
            
            # --- START: NEW DETECTION RATE CALCULATION ---
            num_extracted = len(extracted_state)
            num_matched = 0

            if num_ground_truth > 0 and num_extracted > 0:
                # Use the same logic as OSPA to find matches
                # This ensures consistency between your performance metric and your reward signal
                gt_positions = np.array([obj.get('position', obj.get('mean', [0, 0, 0]))[:3] for obj in ground_truth_at_t])
                est_positions = np.array([obj.get('mean', obj.get('position', [0, 0, 0]))[:3] for obj in extracted_state])
                
                # Calculate distance matrix between all pairs
                diff = gt_positions[:, np.newaxis, :] - est_positions[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff**2, axis=2))
                
                # Use Hungarian algorithm for optimal assignment
                row_indices, col_indices = linear_sum_assignment(distances)
                
                # Count matches below the OSPA cutoff distance
                for r, c in zip(row_indices, col_indices):
                    if distances[r, c] < config.OSPA_CUTOFF_C:
                        num_matched += 1

            # Calculate the true detection rate
            detection_rate = num_matched / num_ground_truth if num_ground_truth > 0 else 1.0 if num_ground_truth == 0 else 0.0
            # --- END: NEW DETECTION RATE CALCULATION ---
            
            # Store the detection rate for the final episode statistics
            if num_ground_truth > 0:  # Only store if we have ground truth objects
                self.last_detection_rate = detection_rate
                self.last_ospa = current_ospa
        else:
            # Use the last calculated values
            detection_rate = getattr(self, 'last_detection_rate', 0.0)
            current_ospa = getattr(self, 'last_ospa', config.OSPA_CUTOFF_C)
            
            # Intermediate reward (scaled down to avoid overwhelming final reward)
            intermediate_reward = -current_ospa * 0.1
            
            # Add detection bonus for successful tracking
            if num_ground_truth > 0:
                # Reward for having tracks when there are ground truth objects
                detection_bonus = min(num_tracks, num_ground_truth) * 10
                intermediate_reward += detection_bonus
                
                # Small penalty for excessive false tracks (but not too harsh)
                if num_tracks > num_ground_truth * 3:
                    excess_tracks = num_tracks - num_ground_truth * 3
                    false_track_penalty = excess_tracks * 2  # Light penalty
                    intermediate_reward -= false_track_penalty
            
            reward = intermediate_reward
            
            # Store OSPA for final episode statistics
            if done:
                self.episode_ospa_scores.append(current_ospa)
                # Final reward gets full weight
                reward = -current_ospa
        
        # Update episode reward
        self.episode_reward += reward
        
        # Prepare info dictionary
        info = {
            'timestep': self.current_time_step + 1,  # Show 1-50 instead of 0-49
            'num_measurements': len(measurements),
            'num_tracks': len(self.estimation_layer.tracks),  # Total tracks in filter
            'num_extracted': len(extracted_state),  # Tracks that pass extraction threshold
            'ospa_distance': current_ospa,
            'detection_rate': detection_rate, # The new, correct metric
            'episode_reward': self.episode_reward
        }
        
        return observation, reward, done, info
    
    def get_episode_statistics(self) -> Dict[str, Any]:
        """
        Get statistics for the current episode.
        
        Returns:
            Dictionary containing episode statistics
        """
        return {
            'episode_reward': self.episode_reward,
            'final_ospa': self.episode_ospa_scores[-1] if self.episode_ospa_scores else None,
            'mean_ospa': np.mean(self.episode_ospa_scores) if self.episode_ospa_scores else None,
            'num_timesteps': self.current_time_step
        }
    
    def get_filter_statistics(self) -> Dict[str, Any]:
        """
        Get detailed statistics about the LMB filter performance.
        
        Returns:
            Dictionary containing filter performance statistics
        """
        # Get current filter state
        current_tracks = self.estimation_layer.tracks
        
        # Calculate basic statistics
        num_tracks = len(current_tracks)
        total_measurements = len(self.unassociated_measurements)
        
        # Calculate track statistics
        track_existences = [track.existence_probability for track in current_tracks]
        avg_existence = np.mean(track_existences) if track_existences else 0.0
        
        # Calculate detection rate (simplified - in real implementation, this would track ground truth)
        # For now, we'll use a heuristic based on track existence probabilities
        high_confidence_tracks = sum(1 for prob in track_existences if prob > 0.7)
        detection_rate = high_confidence_tracks / max(num_tracks, 1)
        
        # Calculate false alarm rate (simplified)
        low_confidence_tracks = sum(1 for prob in track_existences if prob < 0.3)
        false_alarm_rate = low_confidence_tracks / max(num_tracks, 1)
        
        # Calculate track accuracy (simplified - would need ground truth comparison)
        # For now, use average existence probability as a proxy
        track_accuracy = avg_existence
        
        return {
            'num_tracks': num_tracks,
            'num_measurements': total_measurements,
            'detection_rate': detection_rate,
            'false_alarm_rate': false_alarm_rate,
            'track_accuracy': track_accuracy,
            'avg_existence_probability': avg_existence,
            'high_confidence_tracks': high_confidence_tracks,
            'low_confidence_tracks': low_confidence_tracks
        }
    
    def render(self, mode='human'):
        """
        Render the environment (not implemented for this simulation).
        
        Args:
            mode: Rendering mode
        """
        # This could be implemented to visualize the constellation and debris
        pass
    
    def close(self):
        """Close the environment and clean up resources."""
        pass 