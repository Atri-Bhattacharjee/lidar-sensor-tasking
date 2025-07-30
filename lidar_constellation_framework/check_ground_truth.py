#!/usr/bin/env python3
"""
Quick script to examine the ground truth database and check position scales.
"""

import pickle
import numpy as np
import os

def examine_ground_truth():
    """Examine the ground truth database file."""
    
    # Path to the ground truth database
    ground_truth_path = "ground_truth_data/ground_truth_database.pkl"
    
    if not os.path.exists(ground_truth_path):
        print(f"❌ Ground truth database not found at: {ground_truth_path}")
        return
    
    print(f"📁 Loading ground truth database from: {ground_truth_path}")
    
    # Load the database
    with open(ground_truth_path, 'rb') as f:
        ground_truth = pickle.load(f)
    
    print(f"✅ Loaded database with {len(ground_truth)} timesteps")
    
    # Examine first few timesteps
    for timestep in sorted(ground_truth.keys())[:3]:
        objects = ground_truth[timestep]
        print(f"\n📊 Timestep {timestep}: {len(objects)} objects")
        
        if objects:
            # Show first few objects
            for i, obj in enumerate(objects[:3]):
                position = obj.get('position', [0, 0, 0])
                velocity = obj.get('velocity', [0, 0, 0])
                obj_id = obj.get('id', 'unknown')
                
                # Calculate magnitude
                pos_magnitude = np.linalg.norm(position)
                vel_magnitude = np.linalg.norm(velocity)
                
                print(f"  Object {i+1} ({obj_id}):")
                print(f"    Position: {position}")
                print(f"    Position magnitude: {pos_magnitude:.1f} m")
                print(f"    Velocity: {velocity}")
                print(f"    Velocity magnitude: {vel_magnitude:.1f} m/s")
                
                        # Check if position is reasonable for LEO (altitude should be ~200-2000km)
        earth_radius = 6371000  # m
        altitude = pos_magnitude - earth_radius
        
        if altitude < 200000:  # Less than 200km altitude
            print(f"    ⚠️  WARNING: Altitude too low ({altitude/1000:.1f}km)!")
        elif altitude > 2000000:  # More than 2000km altitude
            print(f"    ⚠️  WARNING: Altitude too high ({altitude/1000:.1f}km)!")
        else:
            print(f"    ✅ Altitude looks good ({altitude/1000:.1f}km)")
    
    # Calculate statistics across all timesteps
    print(f"\n📈 Overall Statistics:")
    
    all_positions = []
    all_velocities = []
    
    for timestep, objects in ground_truth.items():
        for obj in objects:
            position = obj.get('position', [0, 0, 0])
            velocity = obj.get('velocity', [0, 0, 0])
            
            all_positions.append(np.linalg.norm(position))
            all_velocities.append(np.linalg.norm(velocity))
    
    if all_positions:
        all_positions = np.array(all_positions)
        all_velocities = np.array(all_velocities)
        
        print(f"  Position magnitudes:")
        print(f"    Min: {all_positions.min():.1f} m")
        print(f"    Max: {all_positions.max():.1f} m")
        print(f"    Mean: {all_positions.mean():.1f} m")
        print(f"    Std: {all_positions.std():.1f} m")
        
        print(f"  Velocity magnitudes:")
        print(f"    Min: {all_velocities.min():.1f} m/s")
        print(f"    Max: {all_velocities.max():.1f} m/s")
        print(f"    Mean: {all_velocities.mean():.1f} m/s")
        print(f"    Std: {all_velocities.std():.1f} m/s")
        
        # Check for reasonable ranges
        earth_radius = 6371000  # meters
        leo_altitude_range = (earth_radius + 200000, earth_radius + 2000000)  # 200km to 2000km
        
        reasonable_positions = np.sum((all_positions >= leo_altitude_range[0]) & 
                                     (all_positions <= leo_altitude_range[1]))
        
        print(f"\n🔍 Position Analysis:")
        print(f"  Earth radius: {earth_radius/1000:.1f} km")
        print(f"  Expected LEO range: {leo_altitude_range[0]/1000:.1f} - {leo_altitude_range[1]/1000:.1f} km")
        print(f"  Positions in LEO range: {reasonable_positions}/{len(all_positions)} ({100*reasonable_positions/len(all_positions):.1f}%)")
        
        if reasonable_positions < len(all_positions) * 0.9:
            print(f"  ⚠️  WARNING: Many positions outside expected LEO range!")
        else:
            print(f"  ✅ Most positions are in expected LEO range")

if __name__ == "__main__":
    examine_ground_truth() 