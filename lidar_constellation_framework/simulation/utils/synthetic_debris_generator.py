"""
Synthetic Debris Generator

This module generates synthetic debris objects based on the actual satellite
orbital parameters, with small variations to create realistic debris scenarios.
"""

import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config
from .constellation_loader import load_constellation_parameters, orbital_elements_to_cartesian


def generate_synthetic_debris_objects() -> List[Dict]:
    """
    Generate 48 synthetic debris objects for testing (reduced from 480 for speed).
    Creates 1 debris object for each satellite instead of 12.
    
    Returns:
        List of debris object dictionaries
    """
    debris_objects = []
    debris_id = 1
    
    # Load constellation parameters
    constellation_params = load_constellation_parameters()
    
    # Create 1 debris object for each satellite (48 total instead of 480)
    for satellite_id, sat_params in constellation_params.items():
        debris_obj = create_debris_from_satellite(sat_params, debris_id, satellite_id)
        debris_objects.append(debris_obj)
        debris_id += 1
    
    return debris_objects


def create_debris_from_satellite(sat_params: Dict[str, float], debris_id: int, satellite_id: str) -> Dict:
    """
    Create a debris object based on satellite orbital parameters with small variations.
    
    Args:
        sat_params: Satellite orbital parameters
        debris_id: Unique ID for the debris object
        satellite_id: ID of the parent satellite
        
    Returns:
        Dictionary containing debris orbital parameters
    """
    # Extract satellite parameters
    sat_semi_major = sat_params['semi_major_axis']
    sat_eccentricity = sat_params['eccentricity']
    sat_inclination = sat_params['inclination']
    sat_raan = sat_params['raan']
    sat_mean_anomaly = sat_params['mean_anomaly']
    
    # Generate variations
    # Semi-major axis variation: ±500 km
    semi_major_variation = np.random.uniform(-500000, 500000)  # ±500 km in meters
    debris_semi_major = sat_semi_major + semi_major_variation
    
    # Keep eccentricity similar (small variation)
    eccentricity_variation = np.random.uniform(-0.0005, 0.0005)
    debris_eccentricity = sat_eccentricity + eccentricity_variation
    debris_eccentricity = max(0.0, debris_eccentricity)  # Ensure non-negative
    
    # Keep inclination similar (small variation)
    inclination_variation = np.random.uniform(-1.0, 1.0)  # ±1 degree
    debris_inclination = sat_inclination + inclination_variation
    
    # RAAN variation: 5-10 degrees
    raan_variation = np.random.uniform(5.0, 10.0) * np.random.choice([-1, 1])  # ±5-10 degrees
    debris_raan = sat_raan + raan_variation
    debris_raan = debris_raan % 360.0  # Normalize to [0, 360)
    
    # Mean anomaly variation: 5-10 degrees
    mean_anomaly_variation = np.random.uniform(5.0, 10.0) * np.random.choice([-1, 1])  # ±5-10 degrees
    debris_mean_anomaly = sat_mean_anomaly + mean_anomaly_variation
    debris_mean_anomaly = debris_mean_anomaly % 360.0  # Normalize to [0, 360)
    
    # Generate random physical properties
    debris_diameter = np.random.uniform(0.1, 2.0)  # 10 cm to 2 m
    debris_mass = np.random.uniform(1.0, 1000.0)  # 1 kg to 1000 kg
    
    return {
        'debris_id': debris_id,
        'semi_major_axis': debris_semi_major,
        'eccentricity': debris_eccentricity,
        'inclination': debris_inclination,
        'raan': debris_raan,
        'mean_anomaly': debris_mean_anomaly,
        'diameter': debris_diameter,
        'mass': debris_mass,
        'parent_satellite': satellite_id
    }


def propagate_debris_orbit(debris_params: Dict, time_seconds: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Propagate debris orbit to a specific time.
    
    Args:
        debris_params: Debris orbital parameters
        time_seconds: Time since epoch in seconds
        
    Returns:
        Tuple of (position, velocity) in meters and m/s
    """
    # Extract orbital elements
    a = debris_params['semi_major_axis']
    e = debris_params['eccentricity']
    i = np.radians(debris_params['inclination'])
    raan = np.radians(debris_params['raan'])
    M0 = np.radians(debris_params['mean_anomaly'])
    
    # Calculate mean motion
    mu = config.GRAVITATIONAL_CONSTANT
    n = np.sqrt(mu / (a**3))  # Mean motion (rad/s)
    
    # Propagate mean anomaly to the specified time
    M = M0 + n * time_seconds
    
    # For near-circular orbits, approximate true anomaly ≈ mean anomaly
    nu = M
    
    # Calculate radius
    r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
    
    # Calculate position in orbital plane
    x_orb = r_mag * np.cos(nu)
    y_orb = r_mag * np.sin(nu)
    z_orb = 0
    
    # Transform to inertial frame
    cos_raan = np.cos(raan)
    sin_raan = np.sin(raan)
    cos_i = np.cos(i)
    sin_i = np.sin(i)
    
    # Position in inertial frame (assuming argument of perigee = 0)
    x = cos_raan * x_orb - sin_raan * cos_i * y_orb
    y = sin_raan * x_orb + cos_raan * cos_i * y_orb
    z = sin_i * y_orb
    
    # Calculate velocity (simplified - circular orbit approximation)
    v_mag = np.sqrt(mu * (2/r_mag - 1/a))
    vx_orb = -v_mag * np.sin(nu)
    vy_orb = v_mag * np.cos(nu)
    vz_orb = 0
    
    # Transform velocity to inertial frame
    vx = cos_raan * vx_orb - sin_raan * cos_i * vy_orb
    vy = sin_raan * vx_orb + cos_raan * cos_i * vy_orb
    vz = sin_i * vy_orb
    
    position = np.array([x, y, z])
    velocity = np.array([vx, vy, vz])
    
    return position, velocity


def create_synthetic_ground_truth_database() -> Dict[int, List[Dict]]:
    """
    Create a ground truth database with propagated positions for all synthetic debris.
    
    Returns:
        Dictionary mapping timesteps to lists of debris states
    """
    print("Generating synthetic debris objects...")
    debris_objects = generate_synthetic_debris_objects()
    
    print("Creating ground truth database...")
    
    # Simulation parameters
    total_timesteps = config.SIMULATION_TIME_STEPS
    episode_duration = config.EPISODE_DURATION_SECONDS
    time_step = episode_duration / total_timesteps
    
    print(f"Total timesteps: {total_timesteps}")
    print(f"Episode duration: {episode_duration} seconds")
    print(f"Time step: {time_step} seconds")
    
    # Initialize ground truth dictionary
    ground_truth_by_timestep = {}
    
    # Propagate each debris object
    for timestep in range(total_timesteps):
        if timestep % 10 == 0:
            print(f"Processing timestep {timestep}/{total_timesteps}")
        
        current_time = timestep * time_step
        timestep_debris = []
        
        for debris in debris_objects:
            # Propagate orbit to current time
            position, velocity = propagate_debris_orbit(debris, current_time)
            
            # Create state vector [Px, Py, Pz, Vx, Vy, Vz]
            state_vector = np.concatenate([position, velocity])
            
            # Create debris state dictionary
            debris_state = {
                'debris_id': debris['debris_id'],
                'diameter': debris['diameter'],
                'mass': debris['mass'],
                'state_vector': state_vector,
                'position': position,
                'velocity': velocity,
                'parent_satellite': debris['parent_satellite']
            }
            
            timestep_debris.append(debris_state)
        
        ground_truth_by_timestep[timestep] = timestep_debris
    
    print(f"Ground truth database created with {len(ground_truth_by_timestep)} timesteps")
    print(f"Total debris states: {sum(len(debris) for debris in ground_truth_by_timestep.values())}")
    
    return ground_truth_by_timestep


def save_synthetic_ground_truth(ground_truth: Dict[int, List[Dict]], filename: str = "synthetic_ground_truth_database.pkl"):
    """
    Save the synthetic ground truth database to a file.
    
    Args:
        ground_truth: Ground truth database
        filename: Output filename
    """
    # Get the ground truth data directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    output_dir = os.path.join(base_dir, "ground_truth_data")
    
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    print(f"Synthetic ground truth database saved to: {output_path}")


def main():
    """Main function to generate and save synthetic ground truth database."""
    print("=" * 60)
    print("Synthetic Debris Ground Truth Generator")
    print("=" * 60)
    
    try:
        # Generate ground truth database
        ground_truth = create_synthetic_ground_truth_database()
        
        # Save to file
        save_synthetic_ground_truth(ground_truth)
        
        print("\n" + "=" * 60)
        print("✓ Synthetic ground truth generation completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error generating synthetic ground truth: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 