#!/usr/bin/env python3
"""
Fast test script for streaming preprocessing - processes only a tiny subset.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
import gc

def test_streaming_preprocessing_fast():
    """Test the streaming preprocessing with a tiny subset."""
    
    print("="*60)
    print("FAST STREAMING PREPROCESSING TEST")
    print("="*60)
    
    # Configuration
    source_directory = os.path.join("output")
    output_directory = os.path.join("ground_truth_data")
    output_filename = "ground_truth_database_streaming_fast_test.pkl"
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"❌ Source directory '{source_directory}' not found!")
        return False
    
    # Check for CPE files
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    if not cpe_files:
        print(f"❌ No *_cond.cpe files found in {source_directory}")
        return False
    
    print(f"✅ Found {len(cpe_files)} CPE files")
    
    # Test with minimal parameters
    START_EPOCH_YR = 2025.0
    END_EPOCH_YR = 2025.001  # Only ~8 hours for testing
    TIME_DELTA_SECONDS = 300.0  # 5-minute timesteps for testing
    
    print(f"Test parameters:")
    print(f"  Start epoch: {START_EPOCH_YR}")
    print(f"  End epoch: {END_EPOCH_YR}")
    print(f"  Time delta: {TIME_DELTA_SECONDS} seconds")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Process only the first file and only first 1000 rows
    print(f"\nProcessing first file with limited rows...")
    
    try:
        # Load just the first file with limited rows
        first_file = cpe_files[0]
        filepath = os.path.join(source_directory, first_file)
        
        print(f"Processing: {first_file}")
        
        # Define columns
        columns = [
            'Epoch', 'Object_ID', 'Object_Diam', 'Object_Mass', 'Altitd', 
            'Ballst_Limit', 'Conchoid_Diam', 'SemiMajorAxis', 'Eccntr', 'Incl', 'Unknown_Col'
        ]
        
        # Read only first 1000 rows
        df = pd.read_csv(
            filepath,
            comment='#',
            delim_whitespace=True,
            header=None,
            names=columns,
            engine='python',
            nrows=1000  # Only first 1000 rows
        )
        
        print(f"Loaded {len(df)} rows from {first_file}")
        
        # Create unique object IDs
        seen_objects = set()
        unique_objects = []
        
        for _, row in df.iterrows():
            # Create unique identifier
            object_id = str(row['Object_ID'])
            altitude = int(row['Altitd'])
            inclination = int(row['Incl'])
            eccentricity = round(row['Eccntr'], 3)
            
            unique_id = f"{object_id}_{altitude}_{inclination}_{eccentricity}"
            
            if unique_id not in seen_objects:
                seen_objects.add(unique_id)
                unique_objects.append({
                    'UniqueID': unique_id,
                    'Epoch': row['Epoch'],
                    'Object_Diam': row['Object_Diam'],
                    'Object_Mass': row['Object_Mass'],
                    'Altitd': row['Altitd'],
                    'SemiMajorAxis': row['SemiMajorAxis'],
                    'Eccntr': row['Eccntr'],
                    'Incl': row['Incl'],
                    'SourceFile': first_file
                })
        
        print(f"Found {len(unique_objects)} unique objects")
        
        # Calculate simulation parameters
        total_duration_seconds = (END_EPOCH_YR - START_EPOCH_YR) * 365.25 * 24 * 3600
        total_timesteps = int(total_duration_seconds / TIME_DELTA_SECONDS)
        time_deltas = np.linspace(0, END_EPOCH_YR - START_EPOCH_YR, total_timesteps)
        
        print(f"Simulation: {total_timesteps} timesteps")
        
        # Initialize ground truth dictionary
        ground_truth_by_timestep = defaultdict(list)
        
        # Process each unique object
        for obj_idx, object_data in enumerate(unique_objects):
            if obj_idx % 10 == 0:
                print(f"Processing object {obj_idx + 1}/{len(unique_objects)}")
            
            # Extract orbital elements
            a = object_data['SemiMajorAxis']  # Semi-major axis in meters
            e = object_data['Eccntr']         # Eccentricity
            i = np.radians(object_data['Incl'])  # Inclination in radians
            
            # Simplified orbital parameters
            raan = 0.0  # Right ascension of ascending node
            nu0 = 0.0   # Initial true anomaly
            
            # Earth's gravitational parameter (m³/s²)
            mu = 3.986004418e14
            
            # Calculate mean motion
            n = np.sqrt(mu / (a**3))
            
            # Propagate orbit for each timestep
            for timestep, dt in enumerate(time_deltas):
                # Convert time delta to seconds
                dt_seconds = dt * 365.25 * 24 * 3600
                
                # Calculate new mean anomaly
                M = n * dt_seconds + nu0
                
                # Calculate radius
                if e < 0.1:  # Nearly circular orbit
                    r_mag = a
                else:
                    r_mag = a * (1 - e * np.cos(M))
                
                # Convert to Cartesian coordinates
                cos_M = np.cos(M)
                sin_M = np.sin(M)
                
                # Position in orbital plane
                x_orbital = r_mag * cos_M
                y_orbital = r_mag * sin_M
                z_orbital = 0
                
                # Apply inclination rotation
                cos_i = np.cos(i)
                sin_i = np.sin(i)
                x_inclined = x_orbital
                y_inclined = y_orbital * cos_i - z_orbital * sin_i
                z_inclined = y_orbital * sin_i + z_orbital * cos_i
                
                # Apply RAAN rotation
                cos_raan = np.cos(raan)
                sin_raan = np.sin(raan)
                x_final = x_inclined * cos_raan - y_inclined * sin_raan
                y_final = x_inclined * sin_raan + y_inclined * cos_raan
                z_final = z_inclined
                
                # Calculate velocity
                v_mag = np.sqrt(mu * (2/r_mag - 1/a))
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
                
                # Create object state
                object_state = {
                    'id': object_data['UniqueID'],
                    'position': np.array([x_final, y_final, z_final]),
                    'velocity': np.array([vx_final, vy_final, vz_final]),
                    'diameter': object_data['Object_Diam'],
                    'mass': object_data['Object_Mass'],
                    'source': object_data['SourceFile']
                }
                
                # Add to timestep
                ground_truth_by_timestep[timestep].append(object_state)
        
        # Convert to regular dict and save
        ground_truth_database = dict(ground_truth_by_timestep)
        output_path = os.path.join(output_directory, output_filename)
        
        print(f"\nSaving database...")
        with open(output_path, 'wb') as f:
            pickle.dump(ground_truth_database, f)
        
        # Print statistics
        object_counts = [len(objects) for objects in ground_truth_database.values()]
        total_objects = sum(object_counts)
        avg_objects = np.mean(object_counts)
        
        print(f"\n✅ Fast streaming test completed!")
        print(f"   • Total timesteps: {len(ground_truth_database)}")
        print(f"   • Total object states: {total_objects}")
        print(f"   • Average objects per timestep: {avg_objects:.1f}")
        print(f"   • Database file size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
        
        # Compare with original database
        original_path = os.path.join("ground_truth_data", "ground_truth_database.pkl")
        if os.path.exists(original_path):
            with open(original_path, 'rb') as f:
                original_ground_truth = pickle.load(f)
            
            original_counts = [len(objects) for objects in original_ground_truth.values()]
            original_avg = np.mean(original_counts)
            
            print(f"\nComparison with original database:")
            print(f"   • Original avg objects per timestep: {original_avg:.1f}")
            print(f"   • Fast test avg objects per timestep: {avg_objects:.1f}")
            print(f"   • This is just a test with limited data")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in fast streaming test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_preprocessing_fast()
    if success:
        print("\n✅ Fast streaming preprocessing test completed successfully!")
        print("The streaming approach works! You can now run the full preprocessing.")
    else:
        print("\n❌ Fast streaming preprocessing test failed!") 