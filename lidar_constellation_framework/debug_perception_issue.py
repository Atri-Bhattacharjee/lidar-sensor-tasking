#!/usr/bin/env python3
"""
Debug script to identify why the perception layer is not generating measurements.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

import config
from environment.perception_layer import PerceptionLayer

def debug_perception_layer():
    """Debug the perception layer step by step."""
    
    print("="*80)
    print("PERCEPTION LAYER DEBUG")
    print("="*80)
    
    # Initialize perception layer
    perception_layer = PerceptionLayer()
    
    print(f"Sensor Parameters:")
    print(f"  FOV: {config.SENSOR_FOV_DEG} degrees")
    print(f"  Max Range: {config.SENSOR_MAX_RANGE_M} m")
    print(f"  Detection Probability Max: {config.PROBABILITY_OF_DETECTION_MAX}")
    print(f"  Constellation Size: {config.NUM_SATELLITES} satellites")
    
    # Check constellation positions
    print(f"\nConstellation Positions: {len(perception_layer.constellation_positions)} satellites")
    for i, pos in enumerate(perception_layer.constellation_positions[:3]):
        print(f"  Satellite {i}: [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")
    
    # Create sample ground truth objects (similar to what we saw in debug output)
    ground_truth = [
        {
            'id': 'test_obj_1',
            'position': np.array([7225600.0, 0.0, 0.0]),  # From debug output
            'velocity': np.array([0.0, 2438.92371426, 7015.46016016]),
            'diameter': 1.0,
            'mass': 100.0
        },
        {
            'id': 'test_obj_2', 
            'position': np.array([7218400.0, 0.0, 0.0]),  # From debug output
            'velocity': np.array([0.0, 2413.1709784, 7028.27577604]),
            'diameter': 1.0,
            'mass': 100.0
        }
    ]
    
    print(f"\nGround Truth Objects: {len(ground_truth)}")
    for i, obj in enumerate(ground_truth):
        pos = obj['position']
        print(f"  Object {i}: [{pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f}]")
    
    # Test different pointing directions
    test_actions = [
        np.zeros((config.NUM_SATELLITES, 2)),  # Point straight ahead
        np.random.uniform(-0.5, 0.5, (config.NUM_SATELLITES, 2)),  # Random pointing
        np.ones((config.NUM_SATELLITES, 2)) * 0.1,  # Small offset
    ]
    
    for test_idx, action in enumerate(test_actions):
        print(f"\n{'='*60}")
        print(f"TEST {test_idx + 1}: Action Type")
        print(f"{'='*60}")
        
        if test_idx == 0:
            print("Action: Point straight ahead (zeros)")
        elif test_idx == 1:
            print("Action: Random pointing")
        else:
            print("Action: Small offset (0.1)")
        
        print(f"Action shape: {action.shape}")
        print(f"First satellite action: [{action[0, 0]:.3f}, {action[0, 1]:.3f}]")
        
        # Generate measurements
        measurements = perception_layer.generate_measurements(ground_truth, action)
        print(f"Measurements generated: {len(measurements)}")
        
        if measurements:
            print("Measurement details:")
            for i, meas in enumerate(measurements[:3]):
                print(f"  Measurement {i+1}:")
                print(f"    Range: {meas.get('range', 'N/A'):.1f} m")
                print(f"    Azimuth: {meas.get('azimuth', 'N/A'):.2f} deg")
                print(f"    Elevation: {meas.get('elevation', 'N/A'):.2f} deg")
                print(f"    Is Clutter: {meas.get('is_clutter', False)}")
                print(f"    Satellite ID: {meas.get('satellite_id', 'N/A')}")
        else:
            print("No measurements generated!")
            
            # Debug why no measurements
            print("\nDebugging why no measurements:")
            
            # Check first satellite and first object
            satellite_pos = perception_layer.constellation_positions[0]
            obj_pos = ground_truth[0]['position']
            pointing_az = action[0, 0] * np.pi
            pointing_el = action[0, 1] * np.pi/4
            
            print(f"  Satellite 0 position: [{satellite_pos[0]:.0f}, {satellite_pos[1]:.0f}, {satellite_pos[2]:.0f}]")
            print(f"  Object 0 position: [{obj_pos[0]:.0f}, {obj_pos[1]:.0f}, {obj_pos[2]:.0f}]")
            print(f"  Pointing angles: az={pointing_az:.3f} rad, el={pointing_el:.3f} rad")
            
            # Calculate relative position
            relative_pos = obj_pos - satellite_pos
            range_to_obj = np.linalg.norm(relative_pos)
            print(f"  Range to object: {range_to_obj:.0f} m")
            print(f"  Max sensor range: {config.SENSOR_MAX_RANGE_M} m")
            print(f"  In range: {range_to_obj <= config.SENSOR_MAX_RANGE_M}")
            
            # Check FOV
            x, y, z = relative_pos
            azimuth = np.arctan2(y, x)
            elevation = np.arcsin(z / range_to_obj)
            print(f"  Object angles: az={azimuth:.3f} rad, el={elevation:.3f} rad")
            
            az_diff = np.abs(azimuth - pointing_az)
            el_diff = np.abs(elevation - pointing_el)
            az_diff = min(az_diff, 2*np.pi - az_diff)
            
            fov_half = np.radians(config.SENSOR_FOV_DEG) / 2
            print(f"  Angular differences: az_diff={az_diff:.3f} rad, el_diff={el_diff:.3f} rad")
            print(f"  FOV half-angle: {fov_half:.3f} rad")
            print(f"  In FOV: {az_diff <= fov_half and el_diff <= fov_half}")
            
            # Check detection probability
            range_scale = 500000.0
            p_detection = config.PROBABILITY_OF_DETECTION_MAX * np.exp(-range_to_obj / range_scale)
            print(f"  Detection probability: {p_detection:.6f}")

def debug_satellite_positions():
    """Debug satellite constellation positions."""
    
    print("\n" + "="*80)
    print("SATELLITE POSITION DEBUG")
    print("="*80)
    
    perception_layer = PerceptionLayer()
    
    # Check if satellites are in reasonable positions
    for i, pos in enumerate(perception_layer.constellation_positions):
        altitude = np.linalg.norm(pos) - config.EARTH_RADIUS_M
        print(f"Satellite {i}: altitude = {altitude/1000:.1f} km")
        
        if i >= 5:  # Only show first 5
            break

def main():
    """Run all debug functions."""
    debug_perception_layer()
    debug_satellite_positions()
    
    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
    print("Based on the output above, we should be able to identify:")
    print("1. Whether satellites are in reasonable positions")
    print("2. Whether objects are within sensor range")
    print("3. Whether objects are within sensor FOV")
    print("4. Whether detection probability is reasonable")

if __name__ == "__main__":
    main() 