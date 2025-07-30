#!/usr/bin/env python3
"""
Quick conversion of ground truth data from kilometers to meters.
"""

import pickle
import numpy as np
import os

def quick_convert():
    """Quickly convert ground truth data from kilometers to meters."""
    
    ground_truth_path = "ground_truth_data/ground_truth_database.pkl"
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth database not found at: {ground_truth_path}")
        return
    
    print(f"📁 Loading ground truth database...")
    
    # Load the database
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    
    print(f"✅ Loaded database with {len(ground_truth)} timesteps")
    
    # Convert from kilometers to meters
    print("🔧 Converting from kilometers to meters...")
    
    # Process in chunks to avoid memory issues
    chunk_size = 50
    timesteps = list(ground_truth.keys())
    
    for i in range(0, len(timesteps), chunk_size):
        chunk_timesteps = timesteps[i:i+chunk_size]
        
        for timestep in chunk_timesteps:
            objects = ground_truth[timestep]
            for obj in objects:
                # Scale position and velocity by 1000 (km to m)
                if 'position' in obj:
                    obj['position'] = np.array(obj['position']) * 1000.0
                if 'velocity' in obj:
                    obj['velocity'] = np.array(obj['velocity']) * 1000.0
        
        print(f"  Processed timesteps {i} to {min(i+chunk_size, len(timesteps))}")
    
    # Save the converted database
    print(f"💾 Saving converted database...")
    with open(ground_truth_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    # Quick verification
    print("\n🔍 Quick verification...")
    sample_timestep = timesteps[0]
    sample_obj = ground_truth[sample_timestep][0]
    
    position = sample_obj.get('position', [0, 0, 0])
    velocity = sample_obj.get('velocity', [0, 0, 0])
    pos_magnitude = np.linalg.norm(position)
    vel_magnitude = np.linalg.norm(velocity)
    
    print(f"  Sample object:")
    print(f"    Position magnitude: {pos_magnitude:.1f} m ({pos_magnitude/1000:.1f} km)")
    print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
    
    earth_radius = 6371000  # m
    altitude = pos_magnitude - earth_radius
    
    if 200000 <= altitude <= 2000000:
        print(f"    ✅ Position now in LEO range! (Altitude: {altitude/1000:.1f} km)")
    else:
        print(f"    ⚠️  Position still outside LEO range (Altitude: {altitude/1000:.1f} km)")
    
    print(f"\n✅ Conversion completed!")

if __name__ == "__main__":
    quick_convert() 