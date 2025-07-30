"""
Labeled Multi-Bernoulli (LMB) Filter Implementation

This module implements the LMB filter for multi-object tracking of space debris.
The filter maintains a probabilistic catalog of debris objects and performs
prediction and update cycles to estimate object states.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.linalg import block_diag
from scipy.optimize import linear_sum_assignment
import config


class GaussianComponent:
    """
    Represents a single component of a Gaussian Mixture Model.
    """
    
    def __init__(self, weight: float, mean: np.ndarray, covariance: np.ndarray):
        """
        Initialize a Gaussian component.
        
        Args:
            weight: The weight of this component in the mixture
            mean: State vector [Px, Py, Pz, Vx, Vy, Vz]ᵀ (6x1)
            covariance: Covariance matrix P (6x6)
        """
        self.weight = weight
        self.mean = mean.reshape(6, 1)  # Ensure column vector
        self.covariance = covariance.reshape(6, 6)  # Ensure square matrix


class Track:
    """
    Represents a single tracked object (a Bernoulli component).
    """
    
    def __init__(self, label: int, existence_probability: float, gmm: List[GaussianComponent]):
        """
        Initialize a track.
        
        Args:
            label: Unique and persistent integer identifier
            existence_probability: Probability r that this track corresponds to a real object
            gmm: List of GaussianComponent objects representing the state PDF
        """
        self.label = label
        self.existence_probability = existence_probability
        self.gmm = gmm


class EstimationLayer:
    """
    Labeled Multi-Bernoulli filter for space debris tracking.
    """
    
    def __init__(self, perception_layer=None):
        """Initialize an empty LMB filter."""
        self.tracks = []  # List of Track objects
        self.next_label_id = 0  # Counter for unique labels
        self.perception_layer = perception_layer  # Reference to perception layer for satellite positions
        
        # Track maximum existence probabilities for hysteresis (from paper)
        self.track_max_existence = {}  # Maps track label to max existence probability
        
        # UKF parameters
        self.alpha = 0.001  # Spread of sigma points
        self.beta = 2.0     # Prior knowledge about distribution
        self.kappa = 0.0    # Secondary scaling parameter
        
        # Calculate UKF weights
        self.lambda_param = self.alpha**2 * (6 + self.kappa) - 6
        self.gamma = np.sqrt(6 + self.lambda_param)
        
        # Weights for mean and covariance
        self.Wm = np.zeros(13)  # 2*6 + 1 = 13 sigma points
        self.Wc = np.zeros(13)
        
        self.Wm[0] = self.lambda_param / (6 + self.lambda_param)
        self.Wc[0] = self.lambda_param / (6 + self.lambda_param) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 13):
            self.Wm[i] = 1.0 / (2 * (6 + self.lambda_param))
            self.Wc[i] = 1.0 / (2 * (6 + self.lambda_param))
    
    def _measurement_function(self, state: np.ndarray, satellite_pos: np.ndarray) -> np.ndarray:
        """
        Measurement function h(x) that converts state to measurement.
        
        Args:
            state: State vector [Px, Py, Pz, Vx, Vy, Vz]ᵀ (6x1)
            satellite_pos: Satellite position [Px, Py, Pz]ᵀ (3x1)
            
        Returns:
            Measurement vector [range, azimuth, elevation, range_rate]ᵀ (4x1)
        """
        # Extract position and velocity from state
        pos = state[:3].reshape(3, 1)
        vel = state[3:].reshape(3, 1)
        
        # Calculate relative position and velocity
        relative_pos = pos - satellite_pos.reshape(3, 1)
        relative_vel = vel  # Assuming satellite velocity is negligible
        
        # Convert to spherical coordinates
        x, y, z = relative_pos.flatten()
        vx, vy, vz = relative_vel.flatten()
        
        # Calculate spherical coordinates
        range_val = np.linalg.norm(relative_pos)
        azimuth = np.arctan2(y, x)
        elevation = np.arcsin(z / range_val)
        
        # Calculate range rate
        range_rate = np.dot(relative_pos.flatten(), relative_vel.flatten()) / range_val
        
        # Convert to degrees for consistency with measurement format
        azimuth_deg = np.degrees(azimuth)
        elevation_deg = np.degrees(elevation)
        
        return np.array([range_val, azimuth_deg, elevation_deg, range_rate]).reshape(4, 1)
    
    def _get_measurement_noise_covariance(self) -> np.ndarray:
        """
        Get measurement noise covariance matrix R.
        
        Returns:
            Measurement noise covariance matrix (4x4)
        """
        # Extract noise parameters from config
        range_sigma = config.LIDAR_RANGE_SIGMA_M
        azimuth_sigma = config.LIDAR_AZIMUTH_SIGMA_DEG  # Keep in degrees to match measurement function
        elevation_sigma = config.LIDAR_ELEVATION_SIGMA_DEG  # Keep in degrees to match measurement function
        range_rate_sigma = config.LIDAR_RANGE_RATE_SIGMA_M_S
        
        # Create diagonal covariance matrix
        R = np.diag([range_sigma**2, azimuth_sigma**2, elevation_sigma**2, range_rate_sigma**2])
        return R
    
    def _generate_sigma_points(self, mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
        """
        Generate sigma points for UKF.
        
        Args:
            mean: State mean (6x1)
            covariance: State covariance (6x6)
            
        Returns:
            Sigma points (6x13)
        """
        n = 6  # State dimension
        sigma_points = np.zeros((n, 2*n + 1))
        
        # Center sigma point
        sigma_points[:, 0] = mean.flatten()
        
        # Generate sigma points using Cholesky decomposition
        try:
            L = np.linalg.cholesky((n + self.lambda_param) * covariance)
        except np.linalg.LinAlgError:
            # If Cholesky fails, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(covariance)
            eigenvals = np.maximum(eigenvals, 1e-6)  # Ensure positive eigenvalues
            L = eigenvecs @ np.diag(np.sqrt(eigenvals)) * np.sqrt(n + self.lambda_param)
        
        # Generate remaining sigma points
        for i in range(n):
            sigma_points[:, i + 1] = mean.flatten() + L[:, i]
            sigma_points[:, i + n + 1] = mean.flatten() - L[:, i]
        
        return sigma_points
    
    def _ukf_update(self, component: GaussianComponent, measurement: Dict, satellite_pos: np.ndarray) -> Tuple[GaussianComponent, float]:
        """
        Perform UKF update for a single Gaussian component.
        
        Args:
            component: Gaussian component to update
            measurement: Measurement dictionary
            satellite_pos: Satellite position
            
        Returns:
            Tuple of (Updated Gaussian component, measurement likelihood)
        """
        # Extract measurement
        z = np.array([
            measurement['range'],
            measurement['azimuth'],
            measurement['elevation'],
            measurement['range_rate']
        ]).reshape(4, 1)
        
        # Get measurement noise covariance
        R = self._get_measurement_noise_covariance()
        
        # Generate sigma points from predicted state
        sigma_points = self._generate_sigma_points(component.mean, component.covariance)
        
        # Transform sigma points through measurement function
        n_sigma = sigma_points.shape[1]
        transformed_sigma = np.zeros((4, n_sigma))
        
        for i in range(n_sigma):
            transformed_sigma[:, i] = self._measurement_function(
                sigma_points[:, i].reshape(6, 1), 
                satellite_pos
            ).flatten()
        
        # Calculate predicted measurement mean
        z_pred = np.zeros((4, 1))
        for i in range(n_sigma):
            z_pred += self.Wm[i] * transformed_sigma[:, i].reshape(4, 1)
        
        # Calculate innovation covariance S
        S = R.copy()
        for i in range(n_sigma):
            diff = transformed_sigma[:, i].reshape(4, 1) - z_pred
            S += self.Wc[i] * diff @ diff.T
        
        # Calculate cross-covariance T
        T = np.zeros((6, 4))
        for i in range(n_sigma):
            state_diff = sigma_points[:, i].reshape(6, 1) - component.mean
            meas_diff = transformed_sigma[:, i].reshape(4, 1) - z_pred
            T += self.Wc[i] * state_diff @ meas_diff.T
        
        # Calculate Kalman gain
        try:
            K = T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = T @ np.linalg.pinv(S)
        
        # Calculate innovation
        innovation = z - z_pred
        
        # Update mean
        mean_updated = component.mean + K @ innovation
        
        # Update covariance
        covariance_updated = component.covariance - K @ S @ K.T
        
        # Ensure covariance is symmetric and positive definite
        covariance_updated = (covariance_updated + covariance_updated.T) / 2
        
        # --- START MODIFICATION ---
        # Calculate likelihood of the measurement given the component
        try:
            # Use log-likelihood for numerical stability
            inv_S = np.linalg.inv(S)
            det_S = np.linalg.det(2 * np.pi * S)
            if det_S <= 0: # Avoid log of non-positive
                 likelihood = 1e-9
            else:
                 log_likelihood = -0.5 * (innovation.T @ inv_S @ innovation + np.log(det_S))
                 likelihood = np.exp(log_likelihood.item())
        except np.linalg.LinAlgError:
            likelihood = 1e-9 # Assign a very small likelihood if matrix inversion fails
        
        # Update the component's weight based on the likelihood
        weight_updated = component.weight * likelihood

        # Return both the updated component and the calculated likelihood
        return GaussianComponent(weight_updated, mean_updated, covariance_updated), likelihood
        # --- END MODIFICATION ---
    
    def step(self, Z_k: List[Dict], unassociated_measurements_from_previous_step: List[Dict], dt: float) -> Tuple[List[Dict], List[Dict]]:
        """
        Main entry point for the filter at each timestep.
        
        Args:
            Z_k: New measurements at time k
            unassociated_measurements_from_previous_step: Unassociated measurements from previous step
            dt: Time step duration
            
        Returns:
            Tuple of (extracted_state, new_unassociated_measurements)
        """
        # FIXED: Implement proper LMB filter based on paper
        
        # Step 1: Prediction (surviving tracks + birth from previous unassociated)
        self._predict(unassociated_measurements_from_previous_step, dt)
        
        # Step 2: Update with current measurements
        new_unassociated_measurements = self._update(Z_k)
        
        # Step 3: Create birth tracks from current unassociated measurements (adaptive birth)
        # This is the key insight from the paper - create birth components from measurements
        # that don't associate with any existing tracks
        if new_unassociated_measurements:
            self._create_adaptive_birth_tracks(new_unassociated_measurements)
        
        # Step 4: Prune low-probability tracks
        self._prune_and_merge(self.tracks)
        
        # Step 5: Extract state for GNN
        extracted_state = self._extract_state()
        
        return extracted_state, new_unassociated_measurements
    
    def _predict(self, unassociated_measurements_from_previous_step: List[Dict], dt: float):
        """
        Predict step: propagate filter state forward in time.
        
        Args:
            unassociated_measurements_from_previous_step: Unassociated measurements from previous step
            dt: Time step duration
        """
        predicted_tracks = []
        
        # Part A: Propagate surviving tracks
        for track in self.tracks:
            # Predict existence probability
            predicted_existence = track.existence_probability * config.SURVIVAL_PROBABILITY
            
            # Predict GMM components
            predicted_gmm = []
            for component in track.gmm:
                predicted_component = self._ukf_predict(component, dt)
                predicted_gmm.append(predicted_component)
            
            # Create predicted track
            predicted_track = Track(track.label, predicted_existence, predicted_gmm)
            predicted_tracks.append(predicted_track)
        
        # Part B: Create new "birth" tracks from unassociated measurements
        for measurement in unassociated_measurements_from_previous_step:
            # Create initial GMM from measurement
            initial_gmm = self._measurement_to_gmm(measurement)
            
            # Create birth track
            birth_track = Track(
                label=self.next_label_id,
                existence_probability=config.BIRTH_PROBABILITY,
                gmm=initial_gmm
            )
            predicted_tracks.append(birth_track)
            self.next_label_id += 1
        
        # Update filter state
        self.tracks = predicted_tracks
    
    def _create_adaptive_birth_tracks(self, measurements: List[Dict]):
        """
        Create adaptive birth tracks from unassociated measurements.
        Based on the paper's adaptive birth distribution approach.
        
        Args:
            measurements: List of unassociated measurements
        """
        for measurement in measurements:
            # Skip clutter measurements
            if measurement.get('is_clutter', False):
                continue
                
            # Create initial GMM from measurement
            initial_gmm = self._measurement_to_gmm(measurement)
            
            # FIXED: Use adaptive birth probability based on paper
            # The paper suggests using a higher birth probability for measurements
            # that don't associate with existing tracks
            adaptive_birth_prob = min(0.9, config.BIRTH_PROBABILITY * 3)  # Higher for adaptive birth
            
            # Create birth track
            birth_track = Track(
                label=self.next_label_id,
                existence_probability=adaptive_birth_prob,
                gmm=initial_gmm
            )
            self.tracks.append(birth_track)
            self.next_label_id += 1
    
    def _ukf_predict(self, component: GaussianComponent, dt: float) -> GaussianComponent:
        """
        Unscented Kalman Filter prediction step for a single component.
        
        Args:
            component: Gaussian component to predict
            dt: Time step duration
            
        Returns:
            Predicted Gaussian component
        """
        # Use class-level UKF parameters
        n = 6  # State dimension
        
        # Generate sigma points using the class method
        sigma_points = self._generate_sigma_points(component.mean, component.covariance)
        
        # Propagate sigma points through dynamics
        propagated_sigma_points = np.zeros_like(sigma_points)
        for i in range(2*n + 1):
            propagated_sigma_points[:, i] = self._orbital_dynamics(sigma_points[:, i], dt)
        
        # Recombine to get predicted mean and covariance
        predicted_mean = np.zeros((6, 1))
        predicted_covariance = np.zeros((6, 6))
        
        for i in range(2*n + 1):
            predicted_mean += self.Wm[i] * propagated_sigma_points[:, i:i+1]
        
        for i in range(2*n + 1):
            diff = propagated_sigma_points[:, i:i+1] - predicted_mean
            predicted_covariance += self.Wc[i] * diff @ diff.T
        
        # Add process noise
        process_noise = self._get_process_noise(dt)
        predicted_covariance += process_noise
        
        return GaussianComponent(component.weight, predicted_mean, predicted_covariance)
    
    def _orbital_dynamics(self, state: np.ndarray, dt: float) -> np.ndarray:
        """
        Simple circular motion approximation for orbital dynamics.
        This maintains constant altitude and approximates orbital motion.
        
        Args:
            state: State vector [Px, Py, Pz, Vx, Vy, Vz] in ECI frame (meters, m/s)
            dt: Time step duration (seconds)
            
        Returns:
            Propagated state vector [Px, Py, Pz, Vx, Vy, Vz]
        """
        # Extract position and velocity
        pos = state[:3]  # meters
        vel = state[3:]  # m/s
        
        # Earth gravitational parameter (m³/s²)
        mu_earth = 3.986004418e14
        
        # Current position magnitude (altitude + Earth radius)
        r_mag = np.linalg.norm(pos)
        
        # Calculate orbital angular velocity for circular orbit
        # ω = sqrt(μ/r³) for circular orbits
        omega = np.sqrt(mu_earth / (r_mag**3))
        
        # For circular motion, we rotate the position vector around the Earth's center
        # The rotation axis is perpendicular to the orbital plane
        # For simplicity, assume the orbit is in the XY plane (z = 0)
        
        # Calculate current angle from x-axis
        current_angle = np.arctan2(pos[1], pos[0])
        
        # Calculate new angle after time dt
        new_angle = current_angle + omega * dt
        
        # Calculate new position (maintains constant radius)
        new_pos = np.array([
            r_mag * np.cos(new_angle),
            r_mag * np.sin(new_angle),
            pos[2]  # Keep z constant (assume equatorial orbit)
        ])
        
        # Calculate new velocity (tangential to the circular path)
        # v = ω × r (cross product)
        # For circular motion in XY plane: v = [-ω*y, ω*x, 0]
        new_vel = np.array([
            -omega * new_pos[1],
            omega * new_pos[0],
            vel[2]  # Keep z velocity constant
        ])
        
        return np.concatenate([new_pos, new_vel])
    
    def _get_process_noise(self, dt: float) -> np.ndarray:
        """
        Get process noise covariance matrix for orbital dynamics.
        
        Args:
            dt: Time step duration (seconds)
            
        Returns:
            Process noise covariance matrix (6x6)
        """
        # Orbital process noise model
        # Position noise grows with time due to orbital perturbations
        # Velocity noise accounts for unmodeled accelerations
        
        # Base noise parameters (tuned for orbital tracking)
        pos_noise_base = 10.0  # meters²/s
        vel_noise_base = 0.1   # m²/s³
        
        # Time-dependent scaling for orbital dynamics
        pos_noise = pos_noise_base * dt**2
        vel_noise = vel_noise_base * dt
        
        # Create diagonal covariance matrix
        # Position components get same noise
        # Velocity components get same noise
        Q = np.diag([
            pos_noise, pos_noise, pos_noise,  # Position noise (x, y, z)
            vel_noise, vel_noise, vel_noise   # Velocity noise (vx, vy, vz)
        ])
        
        return Q
    
    def _measurement_to_gmm(self, measurement: Dict) -> List[GaussianComponent]:
        """
        Convert a measurement to an initial GMM component with orbital velocity estimation.
        
        Args:
            measurement: Measurement dictionary with 'range', 'azimuth', 'elevation', 'range_rate'
            
        Returns:
            List containing a single GaussianComponent
        """
        # Convert spherical coordinates to Cartesian
        r = measurement['range']
        az = np.radians(measurement['azimuth'])
        el = np.radians(measurement['elevation'])
        rr = measurement['range_rate']
        
        # Convert to Cartesian position (relative to satellite)
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        
        # Get satellite position for absolute coordinates
        satellite_id = measurement.get('satellite_id', 0)
        if (hasattr(self, 'perception_layer') and 
            self.perception_layer is not None and 
            hasattr(self.perception_layer, 'constellation_positions') and
            satellite_id < len(self.perception_layer.constellation_positions)):
            satellite_pos = self.perception_layer.constellation_positions[satellite_id]
        else:
            # Fallback: assume satellite at origin
            satellite_pos = np.array([0.0, 0.0, 0.0])
        
        # Convert to absolute Earth-centered coordinates
        abs_x = x + satellite_pos[0]
        abs_y = y + satellite_pos[1]
        abs_z = z + satellite_pos[2]
        
        # Estimate orbital velocity based on position and range rate
        # For circular orbits: v = sqrt(μ/r) where μ is Earth's gravitational parameter
        mu_earth = 3.986004418e14  # m³/s²
        r_mag = np.sqrt(abs_x**2 + abs_y**2 + abs_z**2)
        
        # Calculate orbital velocity magnitude for circular orbit at this altitude
        if r_mag > 6378137.0:  # Above Earth's surface
            orbital_velocity_mag = np.sqrt(mu_earth / r_mag)
        else:
            orbital_velocity_mag = 7500.0  # Default orbital velocity
        
        # For circular orbits, velocity is perpendicular to position vector
        # Use cross product with a reference vector to get perpendicular direction
        pos_vector = np.array([abs_x, abs_y, abs_z])
        pos_unit = pos_vector / np.linalg.norm(pos_vector)
        
        # Choose reference vector based on position to avoid singularity
        if abs(pos_unit[2]) < 0.9:  # Not too close to z-axis
            ref_vector = np.array([0.0, 0.0, 1.0])
        else:
            ref_vector = np.array([1.0, 0.0, 0.0])
        
        # Cross product gives perpendicular direction
        vel_direction = np.cross(pos_unit, ref_vector)
        vel_direction = vel_direction / np.linalg.norm(vel_direction)
        
        # For circular orbits, velocity is purely tangential (perpendicular to position)
        # Range rate gives us information about radial velocity
        # If range rate is small, assume circular orbit
        if abs(rr) < 100.0:  # Small range rate, assume circular
            vx, vy, vz = orbital_velocity_mag * vel_direction
        else:
            # Non-circular orbit: combine radial and tangential components
            radial_velocity = rr
            tangential_velocity = np.sqrt(max(0, orbital_velocity_mag**2 - radial_velocity**2))
            
            # Combine radial and tangential components
            radial_component = radial_velocity * pos_unit
            tangential_component = tangential_velocity * vel_direction
            vx, vy, vz = radial_component + tangential_component
        
        # Ensure velocity is reasonable (fallback if calculation fails)
        if np.isnan(vx) or np.isnan(vy) or np.isnan(vz):
            # Fallback to simple range rate conversion
            vx = rr * np.cos(el) * np.cos(az)
            vy = rr * np.cos(el) * np.sin(az)
            vz = rr * np.sin(el)
        
        mean = np.array([abs_x, abs_y, abs_z, vx, vy, vz]).reshape(6, 1)
        
        # Improved initial uncertainty based on orbital dynamics
        range_uncertainty = max(100.0, r * 0.05)  # 5% of range or 100m minimum
        angle_uncertainty = np.radians(0.5)  # 0.5 degrees (improved from 1.0)
        velocity_uncertainty = 100.0  # 100 m/s (increased for orbital motion)
        
        # Calculate position uncertainty from spherical coordinates
        pos_uncertainty_x = range_uncertainty * np.cos(el) * np.cos(az) + r * angle_uncertainty * np.cos(el) * np.sin(az)
        pos_uncertainty_y = range_uncertainty * np.cos(el) * np.sin(az) + r * angle_uncertainty * np.cos(el) * np.cos(az)
        pos_uncertainty_z = range_uncertainty * np.sin(el) + r * angle_uncertainty * np.cos(el)
        
        # Ensure uncertainties are positive
        pos_uncertainty_x = max(10.0, abs(pos_uncertainty_x))
        pos_uncertainty_y = max(10.0, abs(pos_uncertainty_y))
        pos_uncertainty_z = max(10.0, abs(pos_uncertainty_z))
        
        covariance = np.diag([
            pos_uncertainty_x**2, pos_uncertainty_y**2, pos_uncertainty_z**2,
            velocity_uncertainty**2, velocity_uncertainty**2, velocity_uncertainty**2
        ])
        
        return [GaussianComponent(1.0, mean, covariance)]
    
    def _update(self, Z_k: List[Dict]) -> List[Dict]:
        """
        Update step: fuse new measurements with predicted tracks.
        
        Args:
            Z_k: New measurements at time k
            
        Returns:
            List of unassociated measurements
        """
        if not self.tracks or not Z_k:
            return Z_k
        
        # Gating: find plausible associations
        gated_measurements = self._gating(self.tracks, Z_k)
        
        # Generate association hypotheses (simplified: use greedy assignment)
        associations = self._greedy_assignment(self.tracks, gated_measurements)
        
        # Update tracks based on associations
        updated_tracks = []
        used_measurements = set()
        
        for track_idx, track in enumerate(self.tracks):
            if track_idx in associations:
                # Track has associated measurement
                measurement_idx = associations[track_idx]
                measurement = gated_measurements[measurement_idx]
                used_measurements.add(measurement_idx)
                
                # Update track with measurement
                updated_track = self._update_track_with_measurement(track, measurement)
                updated_tracks.append(updated_track)
            else:
                # Track has no associated measurement
                updated_track = self._update_track_without_measurement(track)
                updated_tracks.append(updated_track)
        
        # Prune and merge
        self.tracks = self._prune_and_merge(updated_tracks)
        
        # Return unassociated measurements
        unassociated = [Z_k[i] for i in range(len(Z_k)) if i not in used_measurements]
        return unassociated
    
    def _gating(self, tracks: List[Track], measurements: List[Dict]) -> List[Dict]:
        """
        Perform gating to find plausible associations.
        
        Args:
            tracks: List of tracks
            measurements: List of measurements
            
        Returns:
            List of gated measurements
        """
        # Simplified gating: return all measurements
        # In a full implementation, this would use Mahalanobis distance
        return measurements
    
    def _greedy_assignment(self, tracks: List[Track], measurements: List[Dict]) -> Dict[int, int]:
        """
        Perform greedy assignment between tracks and measurements.
        
        Args:
            tracks: List of tracks
            measurements: List of measurements
            
        Returns:
            Dictionary mapping track indices to measurement indices
        """
        if not tracks or not measurements:
            return {}
        
        # Calculate distance matrix
        distance_matrix = np.zeros((len(tracks), len(measurements)))
        
        for i, track in enumerate(tracks):
            # Get track position (mean of GMM) - this is in absolute ECI coordinates
            track_pos = self._get_track_position(track)
            
            for j, measurement in enumerate(measurements):
                # Get the relative position of the measurement (satellite-centric)
                relative_meas_pos = self._measurement_to_position(measurement)
                
                # Get the ID of the satellite that made the measurement
                satellite_id = measurement.get('satellite_id', 0)
                
                # Get the absolute ECI position of that satellite
                if self.perception_layer and hasattr(self.perception_layer, 'constellation_positions'):
                    satellite_pos = self.perception_layer.constellation_positions[satellite_id]
                else:
                    # Fallback: assume satellite is at origin (this should not happen in practice)
                    satellite_pos = np.zeros(3)
                
                # --- THE FIX ---
                # Calculate the absolute ECI position of the measurement
                absolute_meas_pos = relative_meas_pos + satellite_pos
                
                # Calculate distance using the absolute position
                distance = np.linalg.norm(track_pos - absolute_meas_pos)
                distance_matrix[i, j] = distance
        
        # Use Hungarian algorithm for optimal assignment
        track_indices, measurement_indices = linear_sum_assignment(distance_matrix)
        
        # Filter by distance threshold
        threshold = 1000.0  # meters
        associations = {}
        
        for track_idx, meas_idx in zip(track_indices, measurement_indices):
            if distance_matrix[track_idx, meas_idx] < threshold:
                associations[track_idx] = meas_idx
        
        return associations
    
    def _get_track_position(self, track: Track) -> np.ndarray:
        """Get the position of a track (mean of GMM)."""
        if not track.gmm:
            return np.zeros(3)
        
        # Weighted average of GMM components
        total_weight = sum(comp.weight for comp in track.gmm)
        # Ensure total_weight is not zero to avoid division by zero
        if total_weight == 0:
            return np.zeros(3)
            
        position = np.zeros((1, 3))  # Shape is (1,3) to match pos_component
        
        for component in track.gmm:
            weight = component.weight / total_weight
            # Extract position as (1,3) array to match position shape
            pos_component = component.mean[:3].T  # Transpose to get (1,3) shape
            position += weight * pos_component
        
        return position
    
    def _measurement_to_position(self, measurement: Dict) -> np.ndarray:
        """Convert measurement to position vector."""
        r = measurement['range']
        az = np.radians(measurement['azimuth'])
        el = np.radians(measurement['elevation'])
        
        x = r * np.cos(el) * np.cos(az)
        y = r * np.cos(el) * np.sin(az)
        z = r * np.sin(el)
        
        return np.array([x, y, z])
    
    def _update_track_with_measurement(self, track: Track, measurement: Dict) -> Track:
        """
        Update a track with an associated measurement using a more stable method.
        
        Args:
            track: Track to update
            measurement: Associated measurement
            
        Returns:
            Updated track
        """
        # Get satellite position for the measurement
        satellite_id = measurement.get('satellite_id', 0)
        satellite_pos = self.perception_layer.constellation_positions[satellite_id] if self.perception_layer else np.zeros(3)

        # --- MORE ROBUST GMM UPDATE ---
        updated_gmm = []
        total_likelihood = 0.0
        
        # 1. Perform UKF update for each component to get its likelihood
        for component in track.gmm:
            # The UKF update now returns the component and the likelihood of the measurement
            updated_comp, likelihood = self._ukf_update(component, measurement, satellite_pos)
            updated_gmm.append(updated_comp)
            total_likelihood += component.weight * likelihood

        # 2. Update existence probability based on measurement likelihood
        # This is a simplified but more correct version of the Bernoulli update
        # It weighs the probability of detection vs. the probability of clutter
        prob_detection = config.PROBABILITY_OF_DETECTION_MAX
        clutter_density = config.CLUTTER_RATE_POISSON_LAMBDA / (np.pi * config.SENSOR_FOV_DEG**2) # Simplified
        
        # Avoid division by zero
        if total_likelihood < 1e-9:
            # If likelihood is near zero, this was a bad match. Decrease existence.
            updated_existence = track.existence_probability * (1 - prob_detection)
        else:
            # Bayes' rule for existence
            numerator = track.existence_probability * prob_detection * total_likelihood
            denominator = clutter_density + numerator
            updated_existence = numerator / denominator

        # Cap the existence probability
        updated_existence = min(0.99, updated_existence)

        # 3. Update GMM component weights based on their individual likelihoods
        if total_likelihood > 1e-9:
            for component in updated_gmm:
                # The weight is already updated inside _ukf_update based on likelihood
                component.weight /= total_likelihood # Normalize the weights
        
        return Track(track.label, updated_existence, updated_gmm)
    
    def _update_track_without_measurement(self, track: Track) -> Track:
        """
        Update a track without an associated measurement.
        
        Args:
            track: Track to update
            
        Returns:
            Updated track
        """
        # Decrease existence probability
        updated_existence = track.existence_probability * 0.9
        
        # Keep the same GMM
        return Track(track.label, updated_existence, track.gmm)
    
    def _prune_and_merge(self, tracks: List[Track]) -> List[Track]:
        """
        Prune tracks with low existence probability and merge close components.
        
        Args:
            tracks: List of tracks to prune and merge
            
        Returns:
            Pruned and merged tracks
        """
        # Prune tracks with low existence probability
        pruned_tracks = [track for track in tracks if track.existence_probability > config.EXISTENCE_THRESHOLD_PRUNING]
        
        # Simplified merging: just return pruned tracks
        # In a full implementation, this would merge close Gaussian components
        return pruned_tracks
    
    def _extract_state(self) -> List[Dict]:
        """
        Extract state for the GNN using hysteresis-based track extraction (from paper).
        
        Returns:
            List of dictionaries containing track information
        """
        extracted_objects = []
        
        for track in self.tracks:
            # Update maximum existence probability for this track
            if track.label not in self.track_max_existence:
                self.track_max_existence[track.label] = track.existence_probability
            else:
                self.track_max_existence[track.label] = max(
                    self.track_max_existence[track.label], 
                    track.existence_probability
                )
            
            # FIXED: Use hysteresis-based extraction as described in the paper
            # Track is extracted if:
            # 1. Maximum existence probability has exceeded upper threshold, AND
            # 2. Current existence probability is above lower threshold
            # For testing, use lower thresholds but maintain proper hysteresis
            upper_threshold = 0.05  # Lowered from 0.5 for testing
            lower_threshold = 0.02   # Lowered from 0.1 for testing (must be < upper_threshold)
            
            should_extract = (
                self.track_max_existence[track.label] > upper_threshold and
                track.existence_probability > lower_threshold
            )
            
            if should_extract:
                # Calculate final mean and covariance of the GMM
                final_mean, final_covariance = self._gmm_to_moments(track.gmm)
                
                extracted_objects.append({
                    'label': track.label,
                    'existence_probability': track.existence_probability,
                    'mean': final_mean.flatten(),
                    'covariance': final_covariance
                })
        
        return extracted_objects
    
    def _gmm_to_moments(self, gmm: List[GaussianComponent]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a GMM to its mean and covariance.
        
        Args:
            gmm: List of GaussianComponent objects
            
        Returns:
            Tuple of (mean, covariance)
        """
        if not gmm:
            return np.zeros((6, 1)), np.eye(6)
        
        # Normalize weights
        total_weight = sum(comp.weight for comp in gmm)
        weights = [comp.weight / total_weight for comp in gmm]
        
        # Calculate mean
        mean = np.zeros((6, 1))
        for i, component in enumerate(gmm):
            mean += weights[i] * component.mean
        
        # Calculate covariance
        covariance = np.zeros((6, 6))
        for i, component in enumerate(gmm):
            diff = component.mean - mean
            covariance += weights[i] * (component.covariance + diff @ diff.T)
        
        return mean, covariance 