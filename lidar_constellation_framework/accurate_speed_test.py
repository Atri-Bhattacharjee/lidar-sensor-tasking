# Accurate speed test using actual preprocessing logic
# Processes only a fraction of a single file to get realistic timing estimates

import os
import pandas as pd
import numpy as np
import pickle
import logging
import time
from typing import Dict, List, Iterator, Tuple
import gc
from collections import defaultdict
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

def stream_cpe_files_test(source_directory: str, max_rows: int = 1000, chunk_size: int = 1000) -> Iterator[pd.DataFrame]:
    """Stream CPE files in chunks - TEST VERSION with row limit."""
    if not os.path.exists(source_directory):
        logger.error(f"Source directory does not exist: {source_directory}")
        return
    
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    
    if not cpe_files:
        logger.error(f"No _cond.cpe files found in {source_directory}")
        return
    
    # Only process the first file
    cpe_file = cpe_files[0]
    logger.info(f"Testing with first file: {cpe_file} (max {max_rows} rows)")
    
    file_path = os.path.join(source_directory, cpe_file)
    
    try:
        # Read file in chunks with row limit
        chunk_iterator = pd.read_csv(
            file_path,
            sep=r'\s+',  # Whitespace separator
            names=define_cpe_columns(),
            chunksize=chunk_size,
            engine='python',
            skiprows=1,  # Skip header row
            on_bad_lines='skip',     # Skip bad lines instead of failing
            nrows=max_rows  # Limit total rows read
        )
        
        for chunk in chunk_iterator:
            # Add source file information
            chunk['SourceFile'] = cpe_file
            yield chunk
            
    except Exception as e:
        logger.error(f"Error processing {cpe_file}: {e}")

def create_unique_object_id(row: pd.Series) -> str:
    """Create a unique identifier for an object."""
    # Use orbital elements to identify unique objects
    # Objects with same semi-major axis, eccentricity, and inclination are the same object
    # Time should NOT be used to distinguish objects - same orbital params = same object
    orbital_signature = f"{row['SemiMajorAxis']:.6f}_{row['Eccntr']:.6f}_{row['Incl.']:.6f}"
    
    # Create unique ID based on orbital signature only
    unique_id = f"obj_{hash(orbital_signature)}"
    return unique_id

def identify_unique_objects_streaming_test(source_directory: str, max_rows: int = 1000) -> Iterator[Tuple[str, Dict]]:
    """Identify unique objects across CPE files using streaming - TEST VERSION."""
    seen_objects = set()
    
    for chunk in stream_cpe_files_test(source_directory, max_rows):
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

def test_preprocessing_speed(source_directory: str, max_rows: int = 1000, 
                           start_epoch: float = 2025.0, end_epoch: float = 2025.25,
                           time_delta_seconds: float = 15.0) -> Dict:
    """Test preprocessing speed with actual logic on limited rows."""
    
    start_time = time.time()
    
    try:
        logger.info("Starting accurate preprocessing speed test...")
        logger.info(f"Max rows to process: {max_rows}")
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
        
        for unique_id, object_data in identify_unique_objects_streaming_test(source_directory, max_rows):
            object_count += 1
            
            if object_count % 5 == 0:  # More frequent updates for test
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
            if object_count % 10 == 0:
                gc.collect()
        
        total_time = time.time() - start_time
        
        logger.info(f"Test completed in {total_time:.2f} seconds")
        logger.info(f"Processed {object_count} unique objects")
        logger.info(f"Generated {total_states} total object states")
        logger.info(f"Timesteps: {len(ground_truth_by_timestep)}")
        
        return {
            'total_time': total_time,
            'object_count': object_count,
            'total_states': total_states,
            'timesteps': len(ground_truth_by_timestep),
            'rows_processed': max_rows
        }
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

def estimate_full_processing_time(test_results: Dict, source_directory: str):
    """Estimate full processing time based on accurate test results."""
    
    if not test_results:
        logger.error("No test results to estimate from")
        return
    
    # Count total files and estimate total rows
    all_cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    total_files = len(all_cpe_files)
    
    # Estimate total rows across all files (rough estimate)
    # Let's assume average file has similar structure
    estimated_rows_per_file = 1000000  # Conservative estimate
    estimated_total_rows = total_files * estimated_rows_per_file
    
    logger.info(f"Total _cond.cpe files: {total_files}")
    logger.info(f"Test processed: {test_results['rows_processed']} rows")
    logger.info(f"Estimated total rows: {estimated_total_rows:,.0f}")
    
    # Calculate scaling factors
    row_ratio = estimated_total_rows / test_results['rows_processed']
    
    # Estimate full processing time
    estimated_total_time = test_results['total_time'] * row_ratio
    
    # Estimate full object counts
    estimated_objects = test_results['object_count'] * row_ratio
    estimated_states = test_results['total_states'] * row_ratio
    
    logger.info("=" * 60)
    logger.info("ACCURATE TIME ESTIMATES:")
    logger.info("=" * 60)
    logger.info(f"Test time: {test_results['total_time']:.2f} seconds")
    logger.info(f"Test objects: {test_results['object_count']}")
    logger.info(f"Test states: {test_results['total_states']}")
    logger.info(f"Row ratio: {row_ratio:.1f}x")
    logger.info("=" * 60)
    logger.info(f"Estimated total objects: {estimated_objects:,.0f}")
    logger.info(f"Estimated total states: {estimated_states:,.0f}")
    logger.info(f"Estimated total time: {estimated_total_time/60:.1f} minutes")
    logger.info(f"Estimated total time: {estimated_total_time/3600:.1f} hours")

def main():
    """Run the accurate speed test."""
    source_directory = os.path.join(os.path.dirname(__file__), "output")
    
    if not os.path.exists(source_directory):
        logger.error(f"Source directory does not exist: {source_directory}")
        return
    
    logger.info("Starting ACCURATE preprocessing speed test...")
    logger.info(f"Source directory: {source_directory}")
    
    # Run test with actual preprocessing logic on 1000 rows from first file
    test_results = test_preprocessing_speed(
        source_directory,
        max_rows=1000,  # Only test 1000 rows from first file
        start_epoch=2025.0,
        end_epoch=2025.25,  # 3 months
        time_delta_seconds=15.0
    )
    
    # Estimate full processing time
    estimate_full_processing_time(test_results, source_directory)

if __name__ == "__main__":
    main() 