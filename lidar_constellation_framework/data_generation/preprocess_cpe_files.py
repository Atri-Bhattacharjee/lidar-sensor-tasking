"""
CPE File Preprocessing Script

This script processes all .cpe event logs to create a complete ground truth database
for the LiDAR constellation simulation. It identifies unique debris objects and
propagates their trajectories forward in time for the entire simulation duration.
"""

import os
import pandas as pd
import numpy as np
import pickle
from astropy.time import Time
import warnings
from typing import Dict, List, Tuple, Any
import sys

# Try to import poliastro, but don't fail if not available
try:
    from poliastro.twobody import Orbit
    from poliastro.bodies import Earth
    POLIASTRO_AVAILABLE = True
except ImportError:
    POLIASTRO_AVAILABLE = False
    print("Warning: poliastro not available. Using simplified orbital propagation.")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add the simulation directory to the path for config access
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'simulation'))
import config


def define_cpe_columns() -> List[str]:
    """
    Define the column names for the .cpe file format.
    
    Returns:
        List of column names for the CPE file format
    """
    return [
        'CPE_No', 'Flux_Contrib', 'Target_Resid', 'Sc', 'Object_Mass', 'Object_Diam',
        'Tr_Lat', 'Veloc', 'Azimuth', 'Elevat', 'Altitd', 'RghtAsc', 'Declin',
        'Srf_Vl', 'Srf_Ang', 'Epoch', 'Ballst_Limit', 'Conchoid_Diam',
        'SemiMajorAxis', 'Eccntr', 'Incl', 'Unknown_Col'  # Added missing column
    ]


def load_and_combine_cpe_files(source_directory: str) -> pd.DataFrame:
    """
    Load and combine all .cpe files into a single DataFrame.
    
    Args:
        source_directory: Directory containing the .cpe files
        
    Returns:
        Combined DataFrame with all CPE data
    """
    print(f"Loading CPE files from: {source_directory}")
    
    # Find all .cpe files with the pattern "*_cond.cpe"
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    print(f"Found {len(cpe_files)} CPE files")
    
    if not cpe_files:
        raise FileNotFoundError(f"No *_cond.cpe files found in {source_directory}")
    
    # Define column names
    cpe_columns = define_cpe_columns()
    
    # Load and combine all files
    all_events_dfs = []
    
    for filename in cpe_files:
        filepath = os.path.join(source_directory, filename)
        print(f"Loading: {filename}")
        
        try:
            # Load the CPE file
            df = pd.read_csv(
                filepath,
                comment='#',
                delim_whitespace=True,
                header=None,
                names=cpe_columns,
                engine='python'
            )
            
            # Add source file information
            df['source_file'] = filename
            
            all_events_dfs.append(df)
            
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            continue
    
    if not all_events_dfs:
        raise ValueError("No CPE files could be loaded successfully")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_events_dfs, ignore_index=True)
    print(f"Combined DataFrame shape: {combined_df.shape}")
    
    return combined_df


def create_unique_object_id(row: pd.Series) -> str:
    """
    Create a unique identifier for an object based on its orbital parameters.
    
    Args:
        row: DataFrame row containing object data
        
    Returns:
        Unique identifier string
    """
    # Create a unique ID based on object characteristics
    # This helps identify the same object across different detections
    unique_parts = [
        f"{row['Object_Diam']:.6f}",
        f"{row['SemiMajorAxis']:.2f}",
        f"{row['Incl']:.2f}",
        f"{row['Eccntr']:.4f}"
    ]
    
    return "_".join(unique_parts)


def identify_unique_objects(combined_df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify unique objects and their initial states.
    
    Args:
        combined_df: Combined DataFrame with all CPE data
        
    Returns:
        DataFrame with unique objects and their initial states
    """
    print("Identifying unique objects...")
    
    # Create unique ID for each row
    combined_df['UniqueID'] = combined_df.apply(create_unique_object_id, axis=1)
    
    # Sort by epoch to ensure we get the earliest detection of each object
    combined_df = combined_df.sort_values('Epoch')
    
    # Keep only the first detection of each unique object
    initial_states_df = combined_df.drop_duplicates(subset=['UniqueID'], keep='first')
    
    print(f"Found {len(initial_states_df)} unique objects out of {len(combined_df)} total detections")
    
    return initial_states_df


def propagate_orbit(initial_elements: Dict[str, float], 
                   initial_epoch: float,
                   time_deltas: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Propagate an orbit using either poliastro (if available) or simplified propagation.
    
    Args:
        initial_elements: Dictionary containing orbital elements
        initial_epoch: Initial epoch in years
        time_deltas: Array of time deltas for propagation
        
    Returns:
        List of (position, velocity) tuples for each timestep
    """
    
    if POLIASTRO_AVAILABLE:
        # Use poliastro for accurate orbital propagation
        try:
            # Create astropy Time object
            initial_time = Time(initial_epoch, format='decimalyear')
            
            # Create poliastro Orbit object with proper units
            from astropy import units as u
            
            orbit = Orbit.from_classical(
                Earth,
                a=initial_elements['SemiMajorAxis'] * u.m,  # Convert to astropy Quantity
                ecc=initial_elements['Eccntr'],
                inc=np.radians(initial_elements['Incl']) * u.rad,  # Convert to astropy Quantity
                raan=0.0 * u.rad,  # Default RAAN with units
                argp=0.0 * u.rad,  # Default argument of perigee with units
                nu=0.0 * u.rad,    # Default true anomaly with units
                epoch=initial_time
            )
            
            # Propagate orbit
            states = []
            for t in time_deltas:
                try:
                    # Propagate to specific time (convert to seconds)
                    t_seconds = t * 365.25 * 24 * 3600  # Convert years to seconds
                    propagated_orbit = orbit.propagate(t_seconds * u.s)
                    
                    # Get position and velocity in Cartesian coordinates
                    r = propagated_orbit.r.to(u.m).value  # Position in meters
                    v = propagated_orbit.v.to(u.m/u.s).value  # Velocity in m/s
                    
                    # Keep in meters for consistency with simulation
                    states.append((r, v))
                except Exception as e:
                    print(f"Warning: Propagation failed at time {t}: {e}")
                    # Handle case where propagation failed
                    states.append((np.array([0, 0, 0]), np.array([0, 0, 0])))
            
            return states
            
        except Exception as e:
            print(f"Warning: poliastro propagation failed: {e}")
            # Fall back to simplified propagation
            pass
    
    # Simplified orbital propagation (fallback)
    return simplified_orbital_propagation(initial_elements, time_deltas)


def simplified_orbital_propagation(initial_elements: Dict[str, float], 
                                  time_deltas: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Simplified orbital propagation using Keplerian motion.
    
    Args:
        initial_elements: Dictionary containing orbital elements
        time_deltas: Array of time deltas for propagation
        
    Returns:
        List of (position, velocity) tuples for each timestep
    """
    # Extract orbital elements
    a = initial_elements['SemiMajorAxis']  # Semi-major axis in meters
    e = initial_elements['Eccntr']         # Eccentricity
    i = np.radians(initial_elements['Incl'])  # Inclination in radians
    raan = 0.0  # Default RAAN
    argp = 0.0  # Default argument of perigee
    nu0 = 0.0   # Default true anomaly
    
    # Earth's gravitational parameter (m³/s²)
    mu = 398600.4418e9
    
    # Calculate orbital period
    T = 2 * np.pi * np.sqrt(a**3 / mu)
    
    # Calculate mean motion
    n = 2 * np.pi / T
    
    states = []
    for t in time_deltas:
        try:
            # Calculate mean anomaly at time t
            M = nu0 + n * t
            
            # Solve Kepler's equation for eccentric anomaly (simplified)
            # For small eccentricities, we can approximate E ≈ M
            E = M  # This is a simplification
            
            # Calculate true anomaly
            nu = 2 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2))
            
            # Calculate radius
            r_mag = a * (1 - e**2) / (1 + e * np.cos(nu))
            
            # Calculate position in orbital plane
            x_orb = r_mag * np.cos(nu)
            y_orb = r_mag * np.sin(nu)
            z_orb = 0
            
            # Transform to inertial frame
            # Rotation matrices for orbital elements
            cos_raan = np.cos(raan)
            sin_raan = np.sin(raan)
            cos_argp = np.cos(argp)
            sin_argp = np.sin(argp)
            cos_i = np.cos(i)
            sin_i = np.sin(i)
            
            # Position in inertial frame
            x = (cos_raan * cos_argp - sin_raan * sin_argp * cos_i) * x_orb + \
                (-cos_raan * sin_argp - sin_raan * cos_argp * cos_i) * y_orb
            y = (sin_raan * cos_argp + cos_raan * sin_argp * cos_i) * x_orb + \
                (-sin_raan * sin_argp + cos_raan * cos_argp * cos_i) * y_orb
            z = sin_argp * sin_i * x_orb + cos_argp * sin_i * y_orb
            
            # Calculate velocity (simplified - circular orbit approximation)
            v_mag = np.sqrt(mu / r_mag)
            vx = -v_mag * np.sin(nu)
            vy = v_mag * np.cos(nu)
            vz = 0
            
            # Transform velocity to inertial frame (simplified)
            vx_inertial = vx
            vy_inertial = vy
            vz_inertial = vz
            
            position = np.array([x, y, z])
            velocity = np.array([vx_inertial, vy_inertial, vz_inertial])
            
            states.append((position, velocity))
            
        except Exception as e:
            print(f"Warning: Simplified propagation failed at time {t}: {e}")
            states.append((np.array([0, 0, 0]), np.array([0, 0, 0])))
    
    return states


def create_ground_truth_database(initial_states_df: pd.DataFrame,
                                start_epoch: float,
                                end_epoch: float,
                                time_delta_seconds: float) -> Dict[int, List[Dict[str, Any]]]:
    """
    Create the ground truth database by propagating all unique objects.
    
    Args:
        initial_states_df: DataFrame with unique objects and initial states
        start_epoch: Start epoch in years
        end_epoch: End epoch in years
        time_delta_seconds: Time step in seconds
        
    Returns:
        Dictionary mapping timesteps to lists of object states
    """
    print("Creating ground truth database...")
    
    # Calculate simulation parameters
    total_duration_seconds = (end_epoch - start_epoch) * 365.25 * 24 * 3600
    total_timesteps = int(total_duration_seconds / time_delta_seconds)
    
    print(f"Simulation duration: {total_duration_seconds:.0f} seconds")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Time step: {time_delta_seconds} seconds")
    
    # Create time deltas array
    time_deltas = np.arange(0, total_duration_seconds + time_delta_seconds, time_delta_seconds)
    
    # Initialize ground truth dictionary
    ground_truth_by_timestep = {}
    
    # Process each unique object
    for idx, row in initial_states_df.iterrows():
        if idx % 100 == 0:
            print(f"Processing object {idx + 1}/{len(initial_states_df)}")
        
        # Extract initial conditions
        initial_elements = {
            'SemiMajorAxis': row['SemiMajorAxis'],
            'Eccntr': row['Eccntr'],
            'Incl': row['Incl']
        }
        
        initial_epoch = row['Epoch']
        unique_id = row['UniqueID']
        object_diameter = row['Object_Diam']
        object_mass = row['Object_Mass']
        
        # Propagate orbit
        propagated_states = propagate_orbit(initial_elements, initial_epoch, time_deltas)
        
        # Add to ground truth database
        for timestep, (position, velocity) in enumerate(propagated_states):
            # Create state vector [Px, Py, Pz, Vx, Vy, Vz]
            state_vector = np.concatenate([position, velocity])
            
            # Create object dictionary
            object_data = {
                'UniqueID': unique_id,
                'diameter': object_diameter,
                'mass': object_mass,
                'state_vector': state_vector,
                'position': position,
                'velocity': velocity
            }
            
            # Add to timestep
            ground_truth_by_timestep.setdefault(timestep, []).append(object_data)
    
    print(f"Ground truth database created with {len(ground_truth_by_timestep)} timesteps")
    
    # Print statistics
    total_objects = sum(len(objects) for objects in ground_truth_by_timestep.values())
    avg_objects_per_timestep = total_objects / len(ground_truth_by_timestep)
    print(f"Total object states: {total_objects}")
    print(f"Average objects per timestep: {avg_objects_per_timestep:.2f}")
    
    return ground_truth_by_timestep


def main():
    """Main function to process CPE files and create ground truth database."""
    print("=" * 60)
    print("CPE File Preprocessing Script")
    print("=" * 60)
    
    # Configuration
    source_directory = "output"  # Directory containing .cpe files
    output_directory = "ground_truth_data"
    output_filename = "ground_truth_database.pkl"
    
    # Simulation time parameters (should match config.py)
    START_EPOCH_YR = 2025.0
    END_EPOCH_YR = 2026.0
    TIME_DELTA_SECONDS = 5.0  # Should match config.SIMULATION_TIME_STEP
    
    print(f"Start epoch: {START_EPOCH_YR}")
    print(f"End epoch: {END_EPOCH_YR}")
    print(f"Time delta: {TIME_DELTA_SECONDS} seconds")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    try:
        # Step 1: Load and combine CPE files
        print("\nStep 1: Loading and combining CPE files...")
        combined_df = load_and_combine_cpe_files(source_directory)
        
        # Step 2: Identify unique objects
        print("\nStep 2: Identifying unique objects...")
        initial_states_df = identify_unique_objects(combined_df)
        
        # Step 3: Create ground truth database
        print("\nStep 3: Creating ground truth database...")
        ground_truth_database = create_ground_truth_database(
            initial_states_df,
            START_EPOCH_YR,
            END_EPOCH_YR,
            TIME_DELTA_SECONDS
        )
        
        # Step 4: Save the database
        print("\nStep 4: Saving ground truth database...")
        output_path = os.path.join(output_directory, output_filename)
        
        with open(output_path, 'wb') as f:
            pickle.dump(ground_truth_database, f)
        
        print(f"Ground truth database saved to: {output_path}")
        
        # Print final statistics
        print("\n" + "=" * 60)
        print("Preprocessing completed successfully!")
        print("=" * 60)
        print(f"Total unique objects: {len(initial_states_df)}")
        print(f"Total timesteps: {len(ground_truth_database)}")
        print(f"Database file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # Sample the database
        sample_timestep = min(ground_truth_database.keys())
        sample_objects = ground_truth_database[sample_timestep]
        print(f"Sample timestep {sample_timestep}: {len(sample_objects)} objects")
        
        if sample_objects:
            sample_object = sample_objects[0]
            print(f"Sample object state vector shape: {sample_object['state_vector'].shape}")
            print(f"Sample object diameter: {sample_object['diameter']} m")
        
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Preprocessing completed successfully!")
    else:
        print("\n❌ Preprocessing failed!")
        sys.exit(1) 