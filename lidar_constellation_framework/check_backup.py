#!/usr/bin/env python3
"""
Check what's in the original backup file to understand the scaling history.
"""

import pickle
import numpy as np
import os

def check_backup():
    """Check the original backup file."""
    
    backup_path = "ground_truth_data/ground_truth_database_backup.pkl"
    
    if not os.path.exists(backup_path):
        print(f"❌ Backup not found at: {backup_path}")
        return
    
    print(f"📁 Loading backup from: {backup_path}")
    
    # Load the backup
    with open(backup_path, 'rb') as f:
        backup_ground_truth = pickle.load(f)
    
    print(f"✅ Loaded backup with {len(backup_ground_truth)} timesteps")
    
    # Check first few objects
    sample_timestep = list(backup_ground_truth.keys())[0]
    sample_objects = backup_ground_truth[sample_timestep][:3]
    
    print(f"\n📊 Backup Analysis (Timestep {sample_timestep}):")
    
    for i, obj in enumerate(sample_objects):
        position = obj.get('position', [0, 0, 0])
        velocity = obj.get('velocity', [0, 0, 0])
        pos_magnitude = np.linalg.norm(position)
        vel_magnitude = np.linalg.norm(velocity)
        
        print(f"  Object {i+1}:")
        print(f"    Position: {position}")
        print(f"    Position magnitude: {pos_magnitude:.1f} m")
        print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
        
        # Check if this looks like the original data
        if pos_magnitude < 1000:
            print(f"    ✅ This looks like the original data (~850m)")
        elif 6e6 <= pos_magnitude <= 8e6:
            print(f"    ✅ This looks like correctly scaled data (~854km)")
        else:
            print(f"    ❌ This looks wrong")

if __name__ == "__main__":
    check_backup() 