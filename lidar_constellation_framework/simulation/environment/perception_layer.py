"""
Perception Layer for LiDAR Constellation Sensor Simulation

This module implements the detailed sensor simulation logic for the LiDAR constellation.
It handles detection probability, measurement noise, and clutter generation.
"""

import numpy as np
from typing import List, Dict, Tuple
import config
from utils.constellation_loader import get_constellation_positions, load_constellation_parameters


class PerceptionLayer:
    """
    Simulates LiDAR sensor measurements for the constellation.
    """
    
    def __init__(self):
        """Initialize the perception layer with sensor parameters."""
        # Store sensor noise parameters from config
        self.range_sigma = config.LIDAR_RANGE_SIGMA_M
        self.azimuth_sigma = np.radians(config.LIDAR_AZIMUTH_SIGMA_DEG)
        self.elevation_sigma = np.radians(config.LIDAR_ELEVATION_SIGMA_DEG)
        self.range_rate_sigma = config.LIDAR_RANGE_RATE_SIGMA_M_S
        self.p_detection_max = config.PROBABILITY_OF_DETECTION_MAX
        self.clutter_lambda = config.CLUTTER_RATE_POISSON_LAMBDA
        
        # Sensor parameters
        self.fov_rad = np.radians(config.SENSOR_FOV_DEG)
        self.max_range = config.SENSOR_MAX_RANGE_M
        
        # Load constellation parameters (positions will be calculated dynamically)
        self.constellation_params = load_constellation_parameters()
        
        # Verify we have the correct number of satellites
        if len(self.constellation_params) != config.NUM_SATELLITES:
            print(f"Warning: Expected {config.NUM_SATELLITES} satellites, but loaded {len(self.constellation_params)}")
            print("This may indicate missing or extra .dia files in the output directory")
    
    def get_constellation_positions_at_time(self, time_seconds: float) -> np.ndarray:
        """
        Get constellation positions at a specific time.
        
        Args:
            time_seconds: Time since epoch in seconds
            
        Returns:
            Array of satellite positions [x, y, z] for each satellite
        """
        return get_constellation_positions(time_seconds)
    
    def generate_measurements(self, ground_truth_at_t: List[Dict], actions: np.ndarray, time_seconds: float = 0.0) -> List[Dict]:
        """
        Generate measurements from all sensors based on ground truth and pointing directions.
        
        Args:
            ground_truth_at_t: List of ground truth objects at time t
            actions: Array of shape (NUM_SATELLITES, 2) with [azimuth, elevation] for each satellite
            time_seconds: Time since epoch in seconds (default: 0.0)
            
        Returns:
            List of all measurements from all sensors
        """
        all_measurements = []
        
        # Get constellation positions at the current time
        constellation_positions = self.get_constellation_positions_at_time(time_seconds)
        
        # Debug output
        if len(ground_truth_at_t) > 0:
            print(f"Perception: Processing {len(ground_truth_at_t)} ground truth objects")
            print(f"Perception: Constellation has {len(constellation_positions)} satellites")
        
        total_in_fov = 0
        total_detected = 0
        
        for satellite_idx in range(config.NUM_SATELLITES):
            # Get satellite position and pointing direction
            satellite_pos = constellation_positions[satellite_idx]
            pointing_az = actions[satellite_idx, 0]  # Normalized [-1, 1]
            pointing_el = actions[satellite_idx, 1]  # Normalized [-1, 1]
            
            # Convert normalized pointing to actual angles
            actual_az = pointing_az * np.pi  # Convert to [-π, π]
            actual_el = pointing_el * np.pi/4  # Convert to [-π/4, π/4]
            
            # Generate measurements for this satellite
            satellite_measurements, in_fov_count, detected_count = self._generate_satellite_measurements_with_debug(
                ground_truth_at_t, satellite_pos, actual_az, actual_el
            )
            
            total_in_fov += in_fov_count
            total_detected += detected_count
            all_measurements.extend(satellite_measurements)
        
        # Debug output
        if len(ground_truth_at_t) > 0:
            print(f"Perception: {total_in_fov} objects in FOV, {total_detected} detected, {len(all_measurements)} measurements generated")
        
        return all_measurements
    
    def _get_satellite_id(self, satellite_pos: np.ndarray, time_seconds: float = 0.0) -> int:
        """
        Get the satellite ID from its position.
        
        Args:
            satellite_pos: Satellite position [x, y, z]
            time_seconds: Time since epoch in seconds (default: 0.0)
            
        Returns:
            Satellite ID (index in constellation)
        """
        # Get constellation positions at the current time
        constellation_positions = self.get_constellation_positions_at_time(time_seconds)
        
        # Find the satellite ID by matching position
        for i, pos in enumerate(constellation_positions):
            if np.allclose(pos, satellite_pos, atol=1.0):  # 1m tolerance
                return i
        
        # Fallback: return 0 if not found
        return 0
    
    def _generate_satellite_measurements(self, ground_truth: List[Dict], 
                                       satellite_pos: np.ndarray, 
                                       pointing_az: float, 
                                       pointing_el: float) -> List[Dict]:
        """
        Generate measurements for a single satellite.
        
        Args:
            ground_truth: List of ground truth objects
            satellite_pos: Satellite position [x, y, z]
            pointing_az: Pointing azimuth angle
            pointing_el: Pointing elevation angle
            
        Returns:
            List of measurements from this satellite
        """
        measurements = []
        
        # Check each ground truth object
        for obj in ground_truth:
            obj_pos = np.array(obj['position'])
            
            # Check if object is in sensor FOV
            if self._is_in_fov(satellite_pos, obj_pos, pointing_az, pointing_el):
                # Apply detection probability
                if self._detect_object(obj, satellite_pos):
                    # Generate noisy measurement
                    measurement = self._generate_noisy_measurement(obj, satellite_pos)
                    measurements.append(measurement)
        
        # No clutter generation - removed completely
        return measurements
    
    def _generate_satellite_measurements_with_debug(self, ground_truth: List[Dict], 
                                                  satellite_pos: np.ndarray, 
                                                  pointing_az: float, 
                                                  pointing_el: float) -> Tuple[List[Dict], int, int]:
        """
        Generate measurements for a single satellite with debug information.
        
        Args:
            ground_truth: List of ground truth objects
            satellite_pos: Satellite position [x, y, z]
            pointing_az: Pointing azimuth angle
            pointing_el: Pointing elevation angle
            
        Returns:
            Tuple of (measurements, in_fov_count, detected_count)
        """
        measurements = []
        in_fov_count = 0
        detected_count = 0
        
        # Check each ground truth object
        for obj in ground_truth:
            obj_pos = np.array(obj['position'])
            
            # Check if object is in sensor FOV
            if self._is_in_fov(satellite_pos, obj_pos, pointing_az, pointing_el):
                in_fov_count += 1
                # Apply detection probability
                if self._detect_object(obj, satellite_pos):
                    detected_count += 1
                    # Generate noisy measurement
                    measurement = self._generate_noisy_measurement(obj, satellite_pos)
                    measurements.append(measurement)
        
        return measurements, in_fov_count, detected_count
    
    def _is_in_fov(self, satellite_pos: np.ndarray, obj_pos: np.ndarray, 
                   pointing_az: float, pointing_el: float) -> bool:
        """
        Check if an object is within the sensor's field of view.
        
        Args:
            satellite_pos: Satellite position
            obj_pos: Object position
            pointing_az: Pointing azimuth
            pointing_el: Pointing elevation
            
        Returns:
            True if object is in FOV, False otherwise
        """
        # Calculate relative position
        relative_pos = obj_pos - satellite_pos
        range_to_obj = np.linalg.norm(relative_pos)
        
        # Check range
        if range_to_obj > self.max_range:
            return False
        
        # Convert to spherical coordinates
        x, y, z = relative_pos
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / range_to_obj)
        
        # Calculate angular differences
        az_diff = np.abs(azimuth - pointing_az)
        el_diff = np.abs(elevation - pointing_el)
        
        # Normalize azimuth difference to [0, π]
        az_diff = min(az_diff, 2*np.pi - az_diff)
        
        # Check if within FOV
        return az_diff <= self.fov_rad/2 and el_diff <= self.fov_rad/2
    
    def _detect_object(self, obj: Dict, satellite_pos: np.ndarray) -> bool:
        """
        Determine if an object is detected based on detection probability model.
        
        Args:
            obj: Ground truth object
            satellite_pos: Satellite position
            
        Returns:
            True if object is detected, False otherwise
        """
        # Use constant detection probability (no range dependence)
        p_detection = self.p_detection_max
        
        # Stochastic detection
        return np.random.random() < p_detection
    
    def _generate_noisy_measurement(self, obj: Dict, satellite_pos: np.ndarray) -> Dict:
        """
        Generate a noisy measurement for a detected object.
        
        Args:
            obj: Ground truth object
            satellite_pos: Satellite position
            
        Returns:
            Dictionary containing noisy measurement
        """
        obj_pos = np.array(obj['position'])
        obj_vel = np.array(obj['velocity'])
        
        # Calculate true relative position and velocity
        relative_pos = obj_pos - satellite_pos
        relative_vel = obj_vel  # Assuming satellite velocity is negligible
        
        # Convert to spherical coordinates
        x, y, z = relative_pos
        vx, vy, vz = relative_vel
        
        # True spherical coordinates
        range_true = np.linalg.norm(relative_pos)
        azimuth_true = np.arctan2(y, x)
        elevation_true = np.arcsin(z / range_true)
        
        # Calculate range rate
        range_rate_true = np.dot(relative_pos, relative_vel) / range_true
        
        # Add measurement noise
        range_noisy = range_true + np.random.normal(0, self.range_sigma)
        azimuth_noisy = azimuth_true + np.random.normal(0, self.azimuth_sigma)
        elevation_noisy = elevation_true + np.random.normal(0, self.elevation_sigma)
        range_rate_noisy = range_rate_true + np.random.normal(0, self.range_rate_sigma)
        
        # Ensure range is positive
        range_noisy = max(0.0, range_noisy)
        
        # Convert back to degrees for consistency
        azimuth_deg = np.degrees(azimuth_noisy)
        elevation_deg = np.degrees(elevation_noisy)
        
        return {
            'range': range_noisy,
            'azimuth': azimuth_deg,
            'elevation': elevation_deg,
            'range_rate': range_rate_noisy,
            'satellite_id': self._get_satellite_id(satellite_pos),  # Get actual satellite ID
            'timestamp': np.random.random()  # Random timestamp within timestep
        } 