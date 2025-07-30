#!/usr/bin/env python3
"""
Simple test to verify the coordinate system fix is working.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

import config
from environment.perception_layer import PerceptionLayer
from environment.estimation_layer import EstimationLayer

def test_simple_coordinate_fix():
    """Test the coordinate system fix with a simple example."""
    
    print("Testing coordinate system fix with simple example...")
    
    # Initialize perception and estimation layers
    perception_layer = PerceptionLayer()
    estimation_layer = EstimationLayer(perception_layer=perception_layer)
    
    # Create a simple measurement (as if it came from a satellite)
    satellite_id = 0
    satellite_pos = perception_layer.constellation_positions[satellite_id]
    
    # Create a ground truth object position
    gt_pos = np.array([7000000.0, 100000.0, 50000.0])  # 7M km from Earth center
    
    # Calculate what the measurement should look like
    relative_pos = gt_pos - satellite_pos
    range_true = np.linalg.norm(relative_pos)
    azimuth_true = np.arctan2(relative_pos[1], relative_pos[0])
    elevation_true = np.arcsin(relative_pos[2] / range_true)
    
    # Create a measurement dictionary
    measurement = {
        'range': range_true,
        'azimuth': np.degrees(azimuth_true),
        'elevation': np.degrees(elevation_true),
        'range_rate': 0.0,
        'satellite_id': satellite_id,
        'timestamp': 0.0
    }
    
    print(f"Ground truth position: {gt_pos}")
    print(f"Satellite position: {satellite_pos}")
    print(f"Relative position: {relative_pos}")
    print(f"Measurement: range={measurement['range']:.1f}m, az={measurement['azimuth']:.1f}°, el={measurement['elevation']:.1f}°")
    
    # Test the coordinate conversion
    relative_meas_pos = estimation_layer._measurement_to_position(measurement)
    print(f"Converted relative position: {relative_meas_pos}")
    
    # Test the fix: convert to absolute position
    absolute_meas_pos = relative_meas_pos + satellite_pos
    print(f"Converted absolute position: {absolute_meas_pos}")
    
    # Check if the fix works
    distance = np.linalg.norm(absolute_meas_pos - gt_pos)
    print(f"Distance to ground truth: {distance:.1f} m")
    
    if distance < 1.0:  # Should be very close
        print("✅ Coordinate system fix working!")
        
        # Test track creation
        print("\nTesting track creation...")
        estimation_layer._create_adaptive_birth_tracks([measurement])
        
        if len(estimation_layer.tracks) > 0:
            track = estimation_layer.tracks[0]
            track_pos = estimation_layer._get_track_position(track)
            track_distance = np.linalg.norm(track_pos - gt_pos)
            print(f"Track position: {track_pos}")
            print(f"Track distance to ground truth: {track_distance:.1f} m")
            
            if track_distance < 1000:  # Should be within 1km
                print("✅ Track creation working!")
                return True
            else:
                print("❌ Track position too far from ground truth!")
                return False
        else:
            print("❌ No tracks created!")
            return False
    else:
        print("❌ Coordinate system fix not working!")
        return False

if __name__ == "__main__":
    success = test_simple_coordinate_fix()
    if success:
        print("\n🎉 The coordinate system fix is working correctly!")
        print("This should resolve the 0% detection rate issue.")
    else:
        print("\n❌ There are still issues with the coordinate system.") 