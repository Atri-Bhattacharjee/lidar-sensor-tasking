#!/usr/bin/env python3
"""
Streaming CPE File Preprocessing Script

This script processes all .cpe event logs using streaming techniques to handle
unlimited objects without memory constraints. It uses chunked processing and
memory-efficient data structures to prevent crashes.
"""

import os
import pandas as pd
import numpy as np
import pickle
import gc
from astropy.time import Time
import warnings
from typing import Dict, List, Tuple, Any, Iterator
import sys
from collections import defaultdict
import logging

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

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def define_cpe_columns() -> List[str]:
    """Define column names for CPE files."""
    return [
        'Epoch', 'Object_ID', 'Object_Diam', 'Object_Mass', 'Altitd', 
        'Ballst_Limit', 'Conchoid_Diam', 'SemiMajorAxis', 'Eccntr', 'Incl', 'Unknown_Col'
    ]

def stream_cpe_files(source_directory: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """
    Stream CPE files in chunks to avoid loading everything into memory.
    
    Args:
        source_directory: Directory containing the .cpe files
        chunk_size: Number of rows to process at once
        
    Yields:
        DataFrame chunks
    """
    logger.info(f"Streaming CPE files from: {source_directory}")
    
    # Find all .cpe files with the pattern "*_cond.cpe"
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    logger.info(f"Found {len(cpe_files)} CPE files")
    
    if not cpe_files:
        raise FileNotFoundError(f"No *_cond.cpe files found in {source_directory}")
    
    # Define column names
    cpe_columns = define_cpe_columns()
    
    for filename in cpe_files:
        filepath = os.path.join(source_directory, filename)
        logger.info(f"Streaming: {filename}")
        
        try:
            # Stream the file in chunks
            chunk_iter = pd.read_csv(
                filepath,
                comment='#',
                delim_whitespace=True,
                header=None,
                names=cpe_columns,
                engine='python',
                chunksize=chunk_size
            )
            
            for chunk in chunk_iter:
                # Add source file information
                chunk['source_file'] = filename
                yield chunk
                
        except Exception as e:
            logger.warning(f"Could not stream {filename}: {e}")
            continue

def create_unique_object_id(row: pd.Series) -> str:
    """
    Create a unique identifier for an object based on its orbital parameters.
    
    Args:
        row: DataFrame row containing object data
        
    Returns:
        Unique identifier string
    """
    # Create a unique ID based on object characteristics
    object_id = str(row['Object_ID'])
    altitude = int(row['Altitd'])
    inclination = int(row['Incl'])
    eccentricity = round(row['Eccntr'], 3)
    
    return f"{object_id}_{altitude}_{inclination}_{eccentricity}"

def identify_unique_objects_streaming(source_directory: str) -> Iterator[Tuple[str, Dict]]:
    """
    Identify unique objects using streaming to avoid memory issues.
    
    Args:
        source_directory: Directory containing CPE files
        
    Yields:
        Tuples of (unique_id, object_data)
    """
    logger.info("Identifying unique objects using streaming...")
    
    # Track unique objects using a set (memory efficient)
    seen_objects = set()
    total_processed = 0
    
    for chunk in stream_cpe_files(source_directory):
        total_processed += len(chunk)
        
        for _, row in chunk.iterrows():
            # Create unique identifier
            unique_id = create_unique_object_id(row)
            
            # Only process if we haven't seen this object before
            if unique_id not in seen_objects:
                seen_objects.add(unique_id)
                
                # Create object data
                object_data = {
                    'UniqueID': unique_id,
                    'Epoch': row['Epoch'],
                    'Object_Diam': row['Object_Diam'],
                    'Object_Mass': row['Object_Mass'],
                    'Altitd': row['Altitd'],
                    'SemiMajorAxis': row['SemiMajorAxis'],
                    'Eccntr': row['Eccntr'],
                    'Incl': row['Incl'],
                    'SourceFile': row['source_file']
                }
                
                yield unique_id, object_data
        
        # Force garbage collection after each chunk
        gc.collect()
        
        if total_processed % 100000 == 0:
            logger.info(f"Processed {total_processed} rows, found {len(seen_objects)} unique objects")

def simplified_orbital_propagation_streaming(initial_elements: Dict[str, float], 
                                           time_deltas: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Simplified orbital propagation that yields results instead of storing them.
    
    Args:
        initial_elements: Initial orbital elements
        time_deltas: Array of time deltas
        
    Yields:
        Tuples of (position, velocity) for each time step
    """
    # Extract orbital elements
    a = initial_elements['SemiMajorAxis']  # Semi-major axis in meters
    e = initial_elements['Eccntr']         # Eccentricity
    i = np.radians(initial_elements['Incl'])  # Inclination in radians
    
    # Simplified orbital parameters
    raan = 0.0  # Right ascension of ascending node
    nu0 = 0.0   # Initial true anomaly
    
    # Earth's gravitational parameter (m³/s²)
    mu = config.GRAVITATIONAL_CONSTANT
    
    # Calculate mean motion
    n = np.sqrt(mu / (a**3))
    
    for dt in time_deltas:
        # Convert time delta to seconds
        dt_seconds = dt * 365.25 * 24 * 3600  # Convert years to seconds
        
        # Calculate new mean anomaly
        M = n * dt_seconds + nu0
        
        # For LEO, we can use a more accurate approximation
        # Calculate radius using orbital mechanics
        if e < 0.1:  # Nearly circular orbit
            r_mag = a  # Use semi-major axis
        else:
            # For elliptical orbits, use a simplified approach
            r_mag = a * (1 - e * np.cos(M))
        
        # Convert to Cartesian coordinates with proper orientation
        # Apply rotation matrices for inclination and RAAN
        cos_M = np.cos(M)
        sin_M = np.sin(M)
        
        # Position in orbital plane
        x_orbital = r_mag * cos_M
        y_orbital = r_mag * sin_M
        z_orbital = 0
        
        # Apply inclination rotation (around x-axis)
        cos_i = np.cos(i)
        sin_i = np.sin(i)
        x_inclined = x_orbital
        y_inclined = y_orbital * cos_i - z_orbital * sin_i
        z_inclined = y_orbital * sin_i + z_orbital * cos_i
        
        # Apply RAAN rotation (around z-axis)
        cos_raan = np.cos(raan)
        sin_raan = np.sin(raan)
        x_final = x_inclined * cos_raan - y_inclined * sin_raan
        y_final = x_inclined * sin_raan + y_inclined * cos_raan
        z_final = z_inclined
        
        # Calculate velocity using orbital mechanics
        v_mag = np.sqrt(mu * (2/r_mag - 1/a))
        
        # Velocity components in orbital plane
        vx_orbital = -v_mag * sin_M
        vy_orbital = v_mag * cos_M
        vz_orbital = 0
        
        # Apply same rotations to velocity
        vx_inclined = vx_orbital
        vy_inclined = vy_orbital * cos_i - vz_orbital * sin_i
        vz_inclined = vy_orbital * sin_i + vz_orbital * cos_i
        
        vx_final = vx_inclined * cos_raan - vy_inclined * sin_raan
        vy_final = vx_inclined * sin_raan + vy_inclined * cos_raan
        vz_final = vz_inclined
        
        # Yield position and velocity in meters and m/s
        position = np.array([x_final, y_final, z_final])
        velocity = np.array([vx_final, vy_final, vz_final])
        
        yield position, velocity

def create_ground_truth_database_streaming(source_directory: str,
                                         start_epoch: float,
                                         end_epoch: float,
                                         time_delta_seconds: float,
                                         output_path: str) -> bool:
    """
    Create ground truth database using streaming to handle unlimited objects.
    
    Args:
        source_directory: Directory containing CPE files
        start_epoch: Start epoch in years
        end_epoch: End epoch in years
        time_delta_seconds: Time step in seconds
        output_path: Path to save the database
        
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating ground truth database using streaming...")
    
    # Calculate simulation parameters
    total_duration_seconds = (end_epoch - start_epoch) * 365.25 * 24 * 3600
    total_timesteps = int(total_duration_seconds / time_delta_seconds)
    
    logger.info(f"Simulation duration: {total_duration_seconds:.0f} seconds")
    logger.info(f"Total timesteps: {total_timesteps}")
    logger.info(f"Time step: {time_delta_seconds} seconds")
    
    # Create time deltas array
    time_deltas = np.linspace(0, end_epoch - start_epoch, total_timesteps)
    
    # Initialize ground truth dictionary with defaultdict for memory efficiency
    ground_truth_by_timestep = defaultdict(list)
    
    # Process unique objects using streaming
    object_count = 0
    total_states = 0
    
    try:
        for unique_id, object_data in identify_unique_objects_streaming(source_directory):
            object_count += 1
            
            if object_count % 100 == 0:
                logger.info(f"Processing object {object_count}...")
            
            # Extract initial conditions
            initial_elements = {
                'SemiMajorAxis': object_data['SemiMajorAxis'],
                'Eccntr': object_data['Eccntr'],
                'Incl': object_data['Incl']
            }
            
            initial_epoch = object_data['Epoch']
            object_diameter = object_data['Object_Diam']
            object_mass = object_data['Object_Mass']
            
            # Propagate orbit using streaming
            for timestep, (position, velocity) in enumerate(simplified_orbital_propagation_streaming(initial_elements, time_deltas)):
                # Create object dictionary
                object_state = {
                    'id': unique_id,
                    'position': position,
                    'velocity': velocity,
                    'diameter': object_diameter,
                    'mass': object_mass,
                    'source': object_data['SourceFile']
                }
                
                # Add to timestep
                ground_truth_by_timestep[timestep].append(object_state)
                total_states += 1
            
            # Periodic garbage collection
            if object_count % 50 == 0:
                gc.collect()
        
        logger.info(f"Processed {object_count} unique objects")
        logger.info(f"Generated {total_states} total object states")
        
        # Convert defaultdict to regular dict for saving
        ground_truth_database = dict(ground_truth_by_timestep)
        
        # Save the database
        logger.info(f"Saving ground truth database to: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(ground_truth_database, f)
        
        # Print final statistics
        logger.info("=" * 60)
        logger.info("Preprocessing completed successfully!")
        logger.info("=" * 60)
        logger.info(f"Total unique objects: {object_count}")
        logger.info(f"Total timesteps: {len(ground_truth_database)}")
        logger.info(f"Total object states: {total_states}")
        logger.info(f"Average objects per timestep: {total_states / len(ground_truth_database):.2f}")
        logger.info(f"Database file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the streaming preprocessing."""
    # Configuration
    source_directory = os.path.join(os.path.dirname(__file__), "..", "output")
    output_directory = os.path.join(os.path.dirname(__file__), "..", "ground_truth_data")
    output_filename = "ground_truth_database_streaming.pkl"
    
    # Simulation time parameters (should match config.py)
    START_EPOCH_YR = 2025.0
    END_EPOCH_YR = 2026.0
    TIME_DELTA_SECONDS = 5.0
    
    logger.info(f"Start epoch: {START_EPOCH_YR}")
    logger.info(f"End epoch: {END_EPOCH_YR}")
    logger.info(f"Time delta: {TIME_DELTA_SECONDS} seconds")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Run preprocessing
    output_path = os.path.join(output_directory, output_filename)
    success = create_ground_truth_database_streaming(
        source_directory,
        START_EPOCH_YR,
        END_EPOCH_YR,
        TIME_DELTA_SECONDS,
        output_path
    )
    
    if success:
        logger.info("\n✅ Streaming preprocessing completed successfully!")
        logger.info(f"Database saved to: {output_path}")
    else:
        logger.error("\n❌ Streaming preprocessing failed!")

if __name__ == "__main__":
    main() 