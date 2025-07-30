# Streaming Preprocessing with 15-second timesteps for faster processing
# This version uses 15-second intervals instead of 5-second for 3x faster preprocessing

import os
import pandas as pd
import numpy as np
import pickle
import logging
from typing import Dict, List, Iterator, Tuple
import gc
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def define_cpe_columns() -> List[str]:
    """Define the expected columns for CPE files."""
    return [
        'CPE_No', 'Flux_Contrib', 'Target_Resid', 'Sc', 'Object_Mass', 'Object_Diam.',
        'Tr.Lat', 'Veloc.', 'Azimuth', 'Elevat.', 'Altitd.', 'RghtAsc', 'Declin.',
        'Srf_Vl', 'Srf_Ang', 'Epoch', 'Ballst_Limit', 'Conchoid_Diam', 
        'SemiMajorAxis', 'Eccntr', 'Incl.'
    ]

def stream_cpe_files(source_directory: str, chunk_size: int = 10000) -> Iterator[pd.DataFrame]:
    """Stream CPE files in chunks to manage memory usage."""
    if not os.path.exists(source_directory):
        logger.error(f"Source directory does not exist: {source_directory}")
        return
    
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    if not cpe_files:
        logger.error(f"No _cond.cpe files found in {source_directory}")
        return
    
    logger.info(f"Found {len(cpe_files)} CPE files")
    
    for cpe_file in cpe_files:
        file_path = os.path.join(source_directory, cpe_file)
        logger.info(f"Processing file: {cpe_file}")
        
        try:
            # Read file in chunks with more flexible parsing
            chunk_iterator = pd.read_csv(
                file_path,
                sep=r'\s+',  # Whitespace separator
                names=define_cpe_columns(),
                chunksize=chunk_size,
                engine='python',
                skiprows=1,  # Skip header row
                on_bad_lines='skip'     # Skip bad lines instead of failing
            )
            
            for chunk in chunk_iterator:
                # Add source file information
                chunk['SourceFile'] = cpe_file
                yield chunk
                
        except Exception as e:
            logger.error(f"Error processing {cpe_file}: {e}")
            continue

def create_unique_object_id(row: pd.Series) -> str:
    """Create a unique identifier for an object."""
    # Use orbital elements to identify unique objects
    # Objects with same semi-major axis, eccentricity, and inclination are the same object
    # Time should NOT be used to distinguish objects - same orbital params = same object
    orbital_signature = f"{row['SemiMajorAxis']:.6f}_{row['Eccntr']:.6f}_{row['Incl.']:.6f}"
    
    # Create unique ID based on orbital signature only
    unique_id = f"obj_{hash(orbital_signature)}"
    return unique_id

def identify_unique_objects_streaming(source_directory: str) -> Iterator[Tuple[str, Dict]]:
    """Identify unique objects across all CPE files using streaming."""
    seen_objects = set()
    
    for chunk in stream_cpe_files(source_directory):
        for _, row in chunk.iterrows():
            try:
                # Create unique identifier
                unique_id = create_unique_object_id(row)
                
                # Skip if already processed
                if unique_id in seen_objects:
                    continue
                
                # Validate required fields
                required_fields = ['SemiMajorAxis', 'Eccntr', 'Incl.', 'Epoch']
                if any(pd.isna(row[field]) for field in required_fields):
                    continue
                
                # Convert to dictionary
                object_data = {
                    'SemiMajorAxis': float(row['SemiMajorAxis']),
                    'Eccntr': float(row['Eccntr']),
                    'Incl': float(row['Incl.']),
                    'Epoch': float(row['Epoch']),
                    'Object_Diam': float(row['Object_Diam.']) if pd.notna(row['Object_Diam.']) else 1.0,
                    'Object_Mass': float(row['Object_Mass']) if pd.notna(row['Object_Mass']) else 1000.0,
                    'SourceFile': row['SourceFile']
                }
                
                seen_objects.add(unique_id)
                yield unique_id, object_data
                
            except Exception as e:
                logger.warning(f"Error processing row: {e}")
                continue

def simplified_orbital_propagation_streaming(initial_elements: Dict[str, float], 
                                           time_deltas: np.ndarray) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """Simplified orbital propagation using Keplerian elements with streaming output."""
    try:
        # Extract orbital elements
        a = initial_elements['SemiMajorAxis']  # Semi-major axis in km
        e = initial_elements['Eccntr']         # Eccentricity
        i = np.radians(initial_elements['Incl'])  # Inclination in radians
        
        # Earth's gravitational parameter (m³/s²)
        mu = 3.986004418e14
        
        # Convert semi-major axis to meters
        a_m = a * 1000.0
        
        # Calculate orbital period
        T = 2 * np.pi * np.sqrt(a_m**3 / mu)
        
        # For each time delta, calculate position and velocity
        for dt in time_deltas:
            # Calculate mean anomaly
            n = 2 * np.pi / T  # Mean motion
            M = n * dt         # Mean anomaly
            
            # Solve Kepler's equation for eccentric anomaly (simplified)
            E = M  # First approximation (valid for small eccentricities)
            
            # Calculate true anomaly
            cos_nu = (np.cos(E) - e) / (1 - e * np.cos(E))
            sin_nu = (np.sqrt(1 - e**2) * np.sin(E)) / (1 - e * np.cos(E))
            nu = np.arctan2(sin_nu, cos_nu)
            
            # Calculate radius
            r = a_m * (1 - e**2) / (1 + e * np.cos(nu))
            
            # Position in orbital plane
            x_orb = r * np.cos(nu)
            y_orb = r * np.sin(nu)
            z_orb = 0
            
            # Transform to ECI (simplified - assuming RAAN=0, arg_perigee=0)
            # This is a simplified transformation
            x_eci = x_orb * np.cos(0) - y_orb * np.sin(0)
            y_eci = x_orb * np.sin(0) + y_orb * np.cos(0)
            z_eci = z_orb
            
            # Apply inclination rotation
            x_final = x_eci
            y_final = y_eci * np.cos(i) - z_eci * np.sin(i)
            z_final = y_eci * np.sin(i) + z_eci * np.cos(i)
            
            position = np.array([x_final, y_final, z_final])
            
            # Calculate velocity (simplified)
            # For circular orbits, v = sqrt(mu/r)
            v_mag = np.sqrt(mu / r)
            velocity = np.array([0, v_mag, 0])  # Simplified velocity vector
            
            yield position, velocity
            
    except Exception as e:
        logger.error(f"Error in orbital propagation: {e}")
        # Return zero position and velocity as fallback
        yield np.zeros(3), np.zeros(3)

def create_ground_truth_database_streaming(source_directory: str,
                                         start_epoch: float,
                                         end_epoch: float,
                                         time_delta_seconds: float,
                                         output_path: str) -> bool:
    """Create ground truth database using streaming processing."""
    try:
        logger.info("Starting streaming ground truth database creation...")
        logger.info(f"Time delta: {time_delta_seconds} seconds")
        logger.info(f"Start epoch: {start_epoch}")
        logger.info(f"End epoch: {end_epoch}")
        
        # Calculate time deltas
        total_duration_years = end_epoch - start_epoch
        total_duration_seconds = total_duration_years * 365.25 * 24 * 3600
        num_timesteps = int(total_duration_seconds / time_delta_seconds)
        
        logger.info(f"Total duration: {total_duration_years:.2f} years")
        logger.info(f"Number of timesteps: {num_timesteps}")
        
        # Create time deltas array
        time_deltas = np.arange(0, total_duration_seconds, time_delta_seconds)
        
        # Initialize ground truth storage
        ground_truth_by_timestep = defaultdict(list)
        
        # Process objects
        object_count = 0
        total_states = 0
        
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
        
        # Avoid division by zero
        if len(ground_truth_database) > 0:
            avg_objects = total_states / len(ground_truth_database)
            logger.info(f"Average objects per timestep: {avg_objects:.2f}")
        else:
            logger.warning("No ground truth data generated - database is empty")
            
        if os.path.exists(output_path):
            logger.info(f"Database file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        else:
            logger.warning("Database file was not created")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the streaming preprocessing with 15-second timesteps."""
    # Configuration
    source_directory = os.path.join(os.path.dirname(__file__), "..", "output")
    output_directory = os.path.join(os.path.dirname(__file__), "..", "ground_truth_data")
    output_filename = "ground_truth_database_15s.pkl"
    
    # Simulation time parameters (15-second timesteps for 3x faster processing)
    START_EPOCH_YR = 2025.0
    END_EPOCH_YR = 2025.25  # 3 months instead of 1 year
    TIME_DELTA_SECONDS = 15.0  # 15 seconds instead of 5 seconds
    
    logger.info(f"Start epoch: {START_EPOCH_YR}")
    logger.info(f"End epoch: {END_EPOCH_YR}")
    logger.info(f"Time delta: {TIME_DELTA_SECONDS} seconds (3x faster than 5s)")
    
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
        logger.info("\n✅ Streaming preprocessing with 15s timesteps completed successfully!")
        logger.info(f"Database saved to: {output_path}")
    else:
        logger.error("\n❌ Streaming preprocessing failed!")

if __name__ == "__main__":
    main() 