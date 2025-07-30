#!/usr/bin/env python3
"""
Debug script to check the ground truth database object count.
"""

import sys
import os
import pickle
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

def check_ground_truth_database():
    """Check the ground truth database for object counts."""
    
    print("="*60)
    print("GROUND TRUTH DATABASE ANALYSIS")
    print("="*60)
    
    # Path to ground truth database
    ground_truth_path = os.path.join("ground_truth_data", "ground_truth_database.pkl")
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth database not found at: {ground_truth_path}")
        return
    
    print(f"Loading ground truth database from: {ground_truth_path}")
    
    try:
        with open(ground_truth_path, 'rb') as f:
            ground_truth = pickle.load(f)
        
        print(f"✅ Successfully loaded ground truth database")
        print(f"   • Total timesteps: {len(ground_truth)}")
        
        # Check object counts for first few timesteps
        print(f"\nObject counts by timestep:")
        for timestep in sorted(ground_truth.keys())[:10]:  # First 10 timesteps
            num_objects = len(ground_truth[timestep])
            print(f"   Timestep {timestep}: {num_objects} objects")
        
        # Calculate statistics
        object_counts = [len(objects) for objects in ground_truth.values()]
        total_objects = sum(object_counts)
        avg_objects = np.mean(object_counts)
        min_objects = min(object_counts)
        max_objects = max(object_counts)
        
        print(f"\nStatistics:")
        print(f"   • Total object states: {total_objects}")
        print(f"   • Average objects per timestep: {avg_objects:.1f}")
        print(f"   • Min objects per timestep: {min_objects}")
        print(f"   • Max objects per timestep: {max_objects}")
        
        # Check if this matches what we saw in the debug output
        if avg_objects < 1000:
            print(f"\n⚠️  WARNING: Only {avg_objects:.1f} objects per timestep!")
            print(f"   This suggests the preprocessing may have limited the number of objects.")
            print(f"   Check if the memory-efficient preprocessing was used with max_objects=1000")
        
        # Sample some objects
        print(f"\nSample objects from timestep 0:")
        if 0 in ground_truth and ground_truth[0]:
            sample_obj = ground_truth[0][0]
            print(f"   Sample object keys: {list(sample_obj.keys())}")
            if 'position' in sample_obj:
                pos = sample_obj['position']
                print(f"   Sample position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
                print(f"   Position magnitude: {np.linalg.norm(pos):.1f} m")
        
    except Exception as e:
        print(f"❌ Error loading ground truth database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_ground_truth_database() 