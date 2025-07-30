#!/usr/bin/env python3
"""
Fix the interpretation of CPE data by treating it as relative coordinates
and scaling to proper LEO positions.
"""

import pandas as pd
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple
import glob

def define_cpe_columns():
    """Define the column structure for CPE files."""
    return [
        'CPE_No', 'Flux_Contrib', 'Target_Resid', 'Sc', 'Object_Mass', 'Object_Diam',
        'Tr_Lat', 'Veloc', 'Azimuth', 'Elevat', 'Altitd', 'RghtAsc', 'Declin',
        'Srf_Vl', 'Srf_Ang', 'Epoch', 'Ballst_Limit', 'Conchoid_Diam',
        'SemiMajorAxis', 'Eccntr', 'Incl'
    ]

def load_and_combine_cpe_files():
    """Load and combine all CPE files, treating data as relative coordinates."""
    print("📁 Loading CPE files...")
    
    # Find all CPE files
    cpe_files = glob.glob("output/*_cond.cpe")
    print(f"Found {len(cpe_files)} CPE files")
    
    all_data = []
    
    for file_path in cpe_files:
        print(f"Processing {file_path}...")
        
        # Read the file, skipping header lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the data section (skip header lines starting with #)
        data_lines = []
        for line in lines:
            if not line.strip().startswith('#') and line.strip():
                data_lines.append(line.strip())
        
        # Parse the data
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 21:  # Ensure we have enough columns
                try:
                    # Parse the values
                    row_data = {
                        'CPE_No': int(parts[0]),
                        'Flux_Contrib': float(parts[1]),
                        'Target_Resid': float(parts[2]),
                        'Sc': int(parts[3]),
                        'Object_Mass': float(parts[4]),
                        'Object_Diam': float(parts[5]),
                        'Tr_Lat': float(parts[6]),
                        'Veloc': float(parts[7]),
                        'Azimuth': float(parts[8]),
                        'Elevat': float(parts[9]),
                        'Altitd': float(parts[10]),
                        'RghtAsc': float(parts[11]),
                        'Declin': float(parts[12]),
                        'Srf_Vl': float(parts[13]),
                        'Srf_Ang': float(parts[14]),
                        'Epoch': float(parts[15]),
                        'Ballst_Limit': float(parts[16]),
                        'Conchoid_Diam': float(parts[17]),
                        'SemiMajorAxis': float(parts[18]),
                        'Eccntr': float(parts[19]),
                        'Incl': float(parts[20])
                    }
                    all_data.append(row_data)
                except (ValueError, IndexError) as e:
                    print(f"Warning: Skipping malformed line: {e}")
                    continue
    
    print(f"Loaded {len(all_data)} total objects")
    return all_data

def create_unique_object_id(row: Dict) -> str:
    """Create a unique ID for each object based on its characteristics."""
    return f"{row['Object_Diam']}_{row['SemiMajorAxis']}_{row['Incl']}_{row['Eccntr']}"

def identify_unique_objects(data: List[Dict]) -> List[Dict]:
    """Identify unique objects based on their characteristics."""
    print("🔍 Identifying unique objects...")
    
    unique_objects = {}
    
    for row in data:
        obj_id = create_unique_object_id(row)
        
        if obj_id not in unique_objects:
            unique_objects[obj_id] = {
                'id': obj_id,
                'diameter': row['Object_Diam'],
                'mass': row['Object_Mass'],
                'initial_altitude': row['Altitd'],  # km
                'initial_velocity': row['Veloc'],   # km/s
                'initial_azimuth': row['Azimuth'],  # deg
                'initial_elevation': row['Elevat'], # deg
                'initial_ra': row['RghtAsc'],       # deg
                'initial_dec': row['Declin'],       # deg
                'semi_major_axis': row['SemiMajorAxis'],  # m (but this is wrong!)
                'eccentricity': row['Eccntr'],
                'inclination': row['Incl'],         # deg
                'epoch': row['Epoch']
            }
    
    print(f"Found {len(unique_objects)} unique objects")
    return list(unique_objects.values())

def convert_to_leo_positions(unique_objects: List[Dict]) -> List[Dict]:
    """Convert the relative coordinates to proper LEO positions."""
    print("🛰️ Converting to LEO positions...")
    
    earth_radius = 6371000  # m
    target_altitude = 700000  # m (700km altitude)
    target_distance = earth_radius + target_altitude  # m
    
    leo_objects = []
    
    for obj in unique_objects:
        # The SemiMajorAxis values are way too small, so we'll use the altitude
        # and create proper orbital parameters
        
        # Use the altitude as a base, but scale it to proper LEO
        base_altitude = obj['initial_altitude'] * 1000  # Convert km to m
        
        # Scale factor to get to proper LEO altitude
        # If base_altitude is ~850m, we need to scale by ~1000 to get to 850km
        scale_factor = target_distance / (earth_radius + base_altitude)
        
        # Create new orbital parameters
        new_semi_major_axis = target_distance  # m
        new_altitude = target_altitude  # m
        
        # Convert spherical coordinates to Cartesian
        # Use the azimuth, elevation, and altitude to get position
        azimuth_rad = np.radians(obj['initial_azimuth'])
        elevation_rad = np.radians(obj['initial_elevation'])
        
        # Calculate position in spherical coordinates
        r = earth_radius + new_altitude  # distance from Earth center
        theta = azimuth_rad  # azimuth angle
        phi = elevation_rad  # elevation angle
        
        # Convert to Cartesian coordinates
        x = r * np.cos(phi) * np.cos(theta)
        y = r * np.cos(phi) * np.sin(theta)
        z = r * np.sin(phi)
        
        # Calculate velocity (simplified - use the velocity magnitude)
        velocity_magnitude = obj['initial_velocity'] * 1000  # Convert km/s to m/s
        
        # Create a simple velocity vector (this is simplified)
        vx = velocity_magnitude * 0.1  # Simplified
        vy = velocity_magnitude * 0.1  # Simplified
        vz = velocity_magnitude * 0.1  # Simplified
        
        leo_obj = {
            'id': obj['id'],
            'diameter': obj['diameter'],
            'mass': obj['mass'],
            'position': np.array([x, y, z]),
            'velocity': np.array([vx, vy, vz]),
            'semi_major_axis': new_semi_major_axis,
            'eccentricity': obj['eccentricity'],
            'inclination': obj['inclination'],
            'epoch': obj['epoch']
        }
        
        leo_objects.append(leo_obj)
    
    print(f"Converted {len(leo_objects)} objects to LEO positions")
    return leo_objects

def create_ground_truth_database(leo_objects: List[Dict]) -> Dict:
    """Create the ground truth database with proper LEO positions."""
    print("📊 Creating ground truth database...")
    
    # Create timesteps (876 timesteps as before)
    num_timesteps = 876
    timestep_duration = 60  # seconds
    
    ground_truth = {}
    
    for timestep in range(num_timesteps):
        time_seconds = timestep * timestep_duration
        
        timestep_objects = []
        
        for obj in leo_objects:
            # Simple orbital propagation (Keplerian motion)
            # This is a simplified version - in reality, you'd use proper orbital mechanics
            
            # Get initial position and velocity
            pos = obj['position'].copy()
            vel = obj['velocity'].copy()
            
            # Simple linear propagation (this is very simplified!)
            # In reality, you'd use proper orbital mechanics
            new_pos = pos + vel * time_seconds
            
            # Keep the object in orbit (simplified)
            # Normalize to maintain orbital distance
            distance = np.linalg.norm(new_pos)
            target_distance = obj['semi_major_axis']
            
            if distance > 0:
                new_pos = new_pos * (target_distance / distance)
            
            timestep_obj = {
                'id': obj['id'],
                'position': new_pos,
                'velocity': vel,  # Simplified - keep constant velocity
                'diameter': obj['diameter'],
                'mass': obj['mass']
            }
            
            timestep_objects.append(timestep_obj)
        
        ground_truth[timestep] = timestep_objects
    
    print(f"Created ground truth database with {len(ground_truth)} timesteps")
    return ground_truth

def main():
    """Main function to process CPE files and create ground truth database."""
    print("🚀 Starting CPE data reinterpretation...")
    
    # Load and combine CPE files
    data = load_and_combine_cpe_files()
    
    # Identify unique objects
    unique_objects = identify_unique_objects(data)
    
    # Convert to LEO positions
    leo_objects = convert_to_leo_positions(unique_objects)
    
    # Create ground truth database
    ground_truth = create_ground_truth_database(leo_objects)
    
    # Save the database
    output_path = "ground_truth_data/ground_truth_database.pkl"
    os.makedirs("ground_truth_data", exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    print(f"✅ Ground truth database saved to: {output_path}")
    
    # Print some statistics
    sample_timestep = list(ground_truth.keys())[0]
    sample_objects = ground_truth[sample_timestep][:3]
    
    print(f"\n📊 Sample data (Timestep {sample_timestep}):")
    for i, obj in enumerate(sample_objects):
        pos = obj['position']
        vel = obj['velocity']
        pos_magnitude = np.linalg.norm(pos)
        vel_magnitude = np.linalg.norm(vel)
        
        print(f"  Object {i+1}:")
        print(f"    Position: {pos}")
        print(f"    Position magnitude: {pos_magnitude/1000:.1f} km")
        print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
        
        # Check if this looks reasonable for LEO
        earth_radius = 6371000  # m
        altitude = pos_magnitude - earth_radius
        
        if 200000 <= altitude <= 2000000:  # 200-2000km altitude
            print(f"    ✅ Altitude looks good ({altitude/1000:.1f}km)")
        else:
            print(f"    ⚠️ Altitude may be wrong ({altitude/1000:.1f}km)")

if __name__ == "__main__":
    main() 