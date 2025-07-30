#!/usr/bin/env python3
"""
Test script to verify the coordinate system fix is working.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

import config
from environment.perception_layer import PerceptionLayer
from environment.estimation_layer import EstimationLayer

def test_coordinate_system_fix():
    """Test that the coordinate system fix is working correctly."""
    
    print("Testing coordinate system fix...")
    
    # Initialize perception and estimation layers
    perception_layer = PerceptionLayer()
    estimation_layer = EstimationLayer(perception_layer=perception_layer)
    
    # Create a simple ground truth object
    ground_truth = [{
        'position': np.array([7000000.0, 100000.0, 50000.0]),  # 7M km from Earth center
        'velocity': np.array([7000.0, 0.0, 0.0]),  # 7 km/s velocity
        'id': 1
    }]
    
    # Create a simple action (point towards the object)
    # Need actions for all satellites (config.NUM_SATELLITES)
    actions = np.zeros((config.NUM_SATELLITES, 2))
    actions[0] = [0.1, 0.05]  # Small azimuth/elevation offsets for first satellite
    
    # Generate measurements
    measurements = perception_layer.generate_measurements(ground_truth, actions)
    
    print(f"Generated {len(measurements)} measurements")
    
    # Filter out clutter measurements
    real_measurements = [m for m in measurements if not m.get('is_clutter', False)]
    clutter_measurements = [m for m in measurements if m.get('is_clutter', False)]
    
    print(f"Real measurements: {len(real_measurements)}")
    print(f"Clutter measurements: {len(clutter_measurements)}")
    
    if not real_measurements:
        print("❌ No real measurements generated! This suggests a detection issue.")
        return False
    
    # Test the coordinate system fix
    for i, measurement in enumerate(real_measurements):
        print(f"\nMeasurement {i}:")
        print(f"  Range: {measurement['range']:.1f} m")
        print(f"  Azimuth: {measurement['azimuth']:.1f}°")
        print(f"  Elevation: {measurement['elevation']:.1f}°")
        print(f"  Satellite ID: {measurement['satellite_id']}")
        
        # Get satellite position
        satellite_id = measurement['satellite_id']
        satellite_pos = perception_layer.constellation_positions[satellite_id]
        print(f"  Satellite position: {satellite_pos}")
        
        # Convert measurement to relative position
        relative_pos = estimation_layer._measurement_to_position(measurement)
        print(f"  Relative position: {relative_pos}")
        
        # Convert to absolute position (the fix)
        absolute_pos = relative_pos + satellite_pos
        print(f"  Absolute position: {absolute_pos}")
        
        # Compare with ground truth
        gt_pos = ground_truth[0]['position']
        distance = np.linalg.norm(absolute_pos - gt_pos)
        print(f"  Distance to ground truth: {distance:.1f} m")
        
        if distance < 1000:  # Should be within 1km
            print("  ✅ Coordinate system fix working!")
        else:
            print("  ❌ Coordinate system fix not working!")
            return False
    
    # Test track creation
    print("\nTesting track creation...")
    estimation_layer._create_adaptive_birth_tracks(real_measurements)
    
    print(f"Created {len(estimation_layer.tracks)} tracks")
    
    for i, track in enumerate(estimation_layer.tracks):
        print(f"  Track {i}: existence_prob={track.existence_probability:.3f}")
        track_pos = estimation_layer._get_track_position(track)
        print(f"    Position: {track_pos}")
        
        # Check if track position is reasonable
        gt_pos = ground_truth[0]['position']
        distance = np.linalg.norm(track_pos - gt_pos)
        print(f"    Distance to ground truth: {distance:.1f} m")
        
        if distance < 1000:
            print("    ✅ Track position is reasonable!")
        else:
            print("    ❌ Track position is too far from ground truth!")
            return False
    
    print("\n🎉 All coordinate system tests passed!")
    return True

if __name__ == "__main__":
    success = test_coordinate_system_fix()
    if success:
        print("\nThe coordinate system fix is working correctly.")
        print("This should resolve the 0% detection rate issue.")
    else:
        print("\nThere are still issues with the coordinate system.") 