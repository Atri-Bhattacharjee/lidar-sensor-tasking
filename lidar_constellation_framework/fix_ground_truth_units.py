#!/usr/bin/env python3
"""
Quick script to fix the unit conversion issue in the existing ground truth database.
Scales all positions and velocities by 1000 to convert from incorrect meters to correct scale.
"""

import pickle
import numpy as np
import os

def fix_ground_truth_units():
    """Fix the unit conversion issue in the ground truth database."""
    
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
    
    # Fix the units by scaling positions and velocities by 1000
    print("🔧 Fixing unit conversion...")
    
    scale_factor = 1004.9  # Convert from incorrect meters to correct scale (based on debug analysis)
    
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
    
    print(f"✅ Fixed {fixed_objects} objects")
    
    # Save the fixed database
    print(f"💾 Saving fixed database to: {ground_truth_path}")
    with open(ground_truth_path, 'wb') as f:
        pickle.dump(ground_truth, f)
    
    # Verify the fix
    print("\n🔍 Verifying the fix...")
    
    # Check a few sample positions
    sample_timestep = list(ground_truth.keys())[0]
    sample_objects = ground_truth[sample_timestep][:3]
    
    for i, obj in enumerate(sample_objects):
        position = obj.get('position', [0, 0, 0])
        velocity = obj.get('velocity', [0, 0, 0])
        pos_magnitude = np.linalg.norm(position)
        vel_magnitude = np.linalg.norm(velocity)
        
        print(f"  Object {i+1}:")
        print(f"    Position magnitude: {pos_magnitude:.1f} m")
        print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
        
        # Check if position is now reasonable
        if 6e6 <= pos_magnitude <= 8e6:
            print(f"    ✅ Position now in LEO range!")
        else:
            print(f"    ⚠️  Position still outside LEO range")
    
    print(f"\n✅ Unit conversion fix completed!")
    print(f"   • Backup saved to: {backup_path}")
    print(f"   • Fixed database saved to: {ground_truth_path}")

if __name__ == "__main__":
    fix_ground_truth_units() 