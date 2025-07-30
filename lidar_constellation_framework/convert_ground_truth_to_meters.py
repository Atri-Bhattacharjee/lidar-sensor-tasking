#!/usr/bin/env python3
"""
Convert ground truth data from kilometers to meters.
"""

import pickle
import numpy as np
import os

def convert_to_meters():
    """Convert ground truth data from kilometers to meters."""
    
    # Path to the ground truth database
    ground_truth_path = "ground_truth_data/ground_truth_database.pkl"
    backup_path = "ground_truth_data/ground_truth_database_backup.pkl"
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth database not found at: {ground_truth_path}")
        return
    
    print(f"📁 Loading ground truth database from: {ground_truth_path}")
    
    # Load the database
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    
    print(f"✅ Loaded database with {len(ground_truth)} timesteps")
    
    # Create backup
    print(f"💾 Creating backup at: {backup_path}")
    with open(backup_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    # Convert from kilometers to meters
    print("🔧 Converting from kilometers to meters...")
    
    scale_factor = 1000.0  # Convert km to m
    
    fixed_objects = 0
    for timestep, objects in ground_truth.items():
        for obj in objects:
            # Scale position
            if 'position' in obj:
                obj['position'] = np.array(obj['position']) * scale_factor
            
            # Scale velocity
            if 'velocity' in obj:
                obj['velocity'] = np.array(obj['velocity']) * scale_factor
            
            fixed_objects += 1
        
        # Print progress every 100 timesteps
        if timestep % 100 == 0:
            print(f"  Processed {timestep} timesteps...")
    
    print(f"✅ Converted {fixed_objects} objects")
    
    # Save the converted database
    print(f"💾 Saving converted database to: {ground_truth_path}")
    with open(ground_truth_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    # Verify the conversion
    print("\n🔍 Verifying the conversion...")
    
    # Check a few sample positions
    sample_timestep = list(ground_truth.keys())[0]
    sample_objects = ground_truth[sample_timestep][:3]
    
    for i, obj in enumerate(sample_objects):
        position = obj.get('position', [0, 0, 0])
        velocity = obj.get('velocity', [0, 0, 0])
        pos_magnitude = np.linalg.norm(position)
        vel_magnitude = np.linalg.norm(velocity)
        
        print(f"  Object {i+1}:")
        print(f"    Position magnitude: {pos_magnitude:.1f} m ({pos_magnitude/1000:.1f} km)")
        print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
        
        # Check if position is now reasonable for LEO
        earth_radius = 6371000  # m
        altitude = pos_magnitude - earth_radius
        
        if 200000 <= altitude <= 2000000:  # 200km to 2000km
            print(f"    ✅ Position now in LEO range! (Altitude: {altitude/1000:.1f} km)")
        else:
            print(f"    ⚠️  Position still outside LEO range (Altitude: {altitude/1000:.1f} km)")
    
    print(f"\n✅ Conversion completed!")
    print(f"   • Backup saved to: {backup_path}")
    print(f"   • Converted database saved to: {ground_truth_path}")

if __name__ == "__main__":
    convert_to_meters() 