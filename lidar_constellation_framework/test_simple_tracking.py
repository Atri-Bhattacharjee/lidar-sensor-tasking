#!/usr/bin/env python3
"""
Simple test script to verify tracking components work correctly.
This will help isolate where the problem is occurring.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from environment.perception_layer import PerceptionLayer
from environment.estimation_layer import EstimationLayer
from utils.ospa import calculate_ospa
import config

def test_perception_layer():
    """Test if the perception layer can detect objects."""
    print("Testing Perception Layer...")
    
    # Create perception layer
    perception = PerceptionLayer()
    
    # Get actual satellite positions to place objects correctly
    satellite_positions = perception.constellation_positions
    print(f"  Satellite constellation: {len(satellite_positions)} satellites")
    print(f"  First satellite position: {satellite_positions[0]}")
    
    # Create objects very close to the first satellite
    first_sat_pos = satellite_positions[0]
    ground_truth = [
        {
            'id': 'test_obj_1',
            'position': first_sat_pos + np.array([1000.0, 0.0, 0.0]),  # 1km in x-direction
            'velocity': np.array([100.0, 200.0, 300.0]),
            'diameter': 1.0,
            'mass': 100.0
        },
        {
            'id': 'test_obj_2', 
            'position': first_sat_pos + np.array([0.0, 1000.0, 0.0]),  # 1km in y-direction
            'velocity': np.array([150.0, 250.0, 350.0]),
            'diameter': 1.5,
            'mass': 150.0
        }
    ]
    
    # Create sensor actions (pointing toward objects)
    sensor_actions = np.zeros((config.NUM_SATELLITES, 2))
    # Point satellites toward the objects
    # Object 1 is at (6871000, 0, 0) - point in +x direction
    # Object 2 is at (0, 6871000, 0) - point in +y direction
    for i in range(config.NUM_SATELLITES):
        if i % 2 == 0:
            # Point toward object 1 (x-axis)
            sensor_actions[i, 0] = 0.0  # Azimuth 0 (pointing along x-axis)
            sensor_actions[i, 1] = 0.0  # Elevation 0 (horizontal)
        else:
            # Point toward object 2 (y-axis)
            sensor_actions[i, 0] = 0.5  # Azimuth π/2 (pointing along y-axis)
            sensor_actions[i, 1] = 0.0  # Elevation 0 (horizontal)
    
    # Generate measurements
    measurements = perception.generate_measurements(ground_truth, sensor_actions)
    
    print(f"  Ground truth objects: {len(ground_truth)}")
    print(f"  Object positions relative to first satellite:")
    for i, obj in enumerate(ground_truth):
        rel_pos = obj['position'] - first_sat_pos
        print(f"    Object {i+1}: {rel_pos}")
    print(f"  Measurements generated: {len(measurements)}")
    if len(measurements) > 0:
        print(f"  First measurement: {measurements[0]}")
        print("  ✓ Perception layer working")
    else:
        print("  ✗ Perception layer not generating measurements")
    
    return len(measurements) > 0

def test_estimation_layer():
    """Test if the estimation layer can track objects."""
    print("\nTesting Estimation Layer...")
    
    # Create perception layer for satellite positions
    perception = PerceptionLayer()
    
    # Create estimation layer with perception layer reference
    estimation = EstimationLayer(perception_layer=perception)
    
    # Create simple measurements (NOT clutter - these should create tracks)
    measurements = [
        {
            'range': 1000.0,
            'azimuth': 0.5,
            'elevation': 0.3,
            'range_rate': 100.0,
            'satellite_id': 0,  # Add satellite ID
            'timestamp': 0.1    # Add timestamp
        },
        {
            'range': 2000.0,
            'azimuth': 0.7,
            'elevation': 0.4,
            'range_rate': 150.0,
            'satellite_id': 1,  # Add satellite ID
            'timestamp': 0.2    # Add timestamp
        }
    ]
    
    # Step the estimation layer
    dt = 1.0  # 1 second time step
    extracted_state, unassociated = estimation.step(measurements, [], dt)
    
    print(f"  Input measurements: {len(measurements)}")
    print(f"  Extracted tracks: {len(extracted_state)}")
    print(f"  Unassociated measurements: {len(unassociated)}")
    print(f"  Total tracks in filter: {len(estimation.tracks)}")
    
    # Debug: Check if birth tracks were created
    if len(estimation.tracks) > 0:
        print(f"  Track existence probabilities: {[t.existence_probability for t in estimation.tracks]}")
        print(f"  Track labels: {[t.label for t in estimation.tracks]}")
    
    if len(extracted_state) > 0:
        print(f"  First track: {extracted_state[0]}")
        print("  ✓ Estimation layer working")
    else:
        print("  ✗ Estimation layer not creating tracks")
    
    return len(extracted_state) > 0

def test_ospa_calculation():
    """Test if OSPA calculation works correctly."""
    print("\nTesting OSPA Calculation...")
    
    # Create ground truth and estimated objects
    ground_truth = [
        {
            'id': 'gt_1',
            'position': np.array([1000.0, 2000.0, 3000.0]),
            'velocity': np.array([100.0, 200.0, 300.0]),
            'diameter': 1.0,
            'mass': 100.0
        }
    ]
    
    estimated = [
        {
            'mean': np.array([1005.0, 2005.0, 3005.0, 100.0, 200.0, 300.0]),
            'covariance': np.eye(6) * 10.0,
            'existence_probability': 0.9
        }
    ]
    
    # Calculate OSPA
    ospa_distance = calculate_ospa(ground_truth, estimated, 1000.0, 1.0)
    
    print(f"  Ground truth objects: {len(ground_truth)}")
    print(f"  Estimated objects: {len(estimated)}")
    print(f"  OSPA distance: {ospa_distance:.2f}")
    
    if ospa_distance < 1000.0:
        print("  ✓ OSPA calculation working")
        return True
    else:
        print("  ✗ OSPA calculation not working")
        return False

def test_end_to_end():
    """Test the complete pipeline."""
    print("\nTesting End-to-End Pipeline...")
    
    # Create components
    perception = PerceptionLayer()
    estimation = EstimationLayer(perception_layer=perception)
    
    # Get actual satellite positions to place objects correctly
    satellite_positions = perception.constellation_positions
    first_sat_pos = satellite_positions[0]
    
    # Create ground truth (same as working perception test)
    ground_truth = [
        {
            'id': 'test_obj_1',
            'position': first_sat_pos + np.array([1000.0, 0.0, 0.0]),  # 1km in x-direction
            'velocity': np.array([100.0, 200.0, 300.0]),
            'diameter': 1.0,
            'mass': 100.0
        },
        {
            'id': 'test_obj_2', 
            'position': first_sat_pos + np.array([0.0, 1000.0, 0.0]),  # 1km in y-direction
            'velocity': np.array([150.0, 250.0, 350.0]),
            'diameter': 1.5,
            'mass': 150.0
        }
    ]
    
    # Create sensor actions (same as working perception test)
    sensor_actions = np.zeros((config.NUM_SATELLITES, 2))
    for i in range(config.NUM_SATELLITES):
        if i % 2 == 0:
            # Point toward object 1 (x-axis)
            sensor_actions[i, 0] = 0.0  # Azimuth 0
            sensor_actions[i, 1] = 0.0  # Elevation 0
        else:
            # Point toward object 2 (y-axis)
            sensor_actions[i, 0] = 0.5  # Azimuth π/2
            sensor_actions[i, 1] = 0.0  # Elevation 0
    
    # Generate measurements
    measurements = perception.generate_measurements(ground_truth, sensor_actions)
    
    if len(measurements) == 0:
        print("  ✗ No measurements generated")
        return False
    
    # Track objects
    dt = 1.0
    extracted_state, unassociated = estimation.step(measurements, [], dt)
    
    if len(extracted_state) == 0:
        print("  ✗ No tracks created")
        return False
    
    # Calculate OSPA
    print(f"  Ground truth objects: {len(ground_truth)}")
    print(f"  Estimated objects: {len(extracted_state)}")
    if len(ground_truth) > 0 and len(extracted_state) > 0:
        print(f"  Ground truth position: {ground_truth[0]['position']}")
        print(f"  Estimated position: {extracted_state[0]['mean'][:3]}")
    
    # Debug: Check measurement details and coordinate consistency
    if len(measurements) > 0:
        print(f"  First measurement: {measurements[0]}")
        satellite_id = measurements[0].get('satellite_id', 0)
        if hasattr(estimation, 'perception_layer'):
            satellite_pos = estimation.perception_layer.constellation_positions[satellite_id]
            print(f"  Satellite {satellite_id} position: {satellite_pos}")
            
            # Verify coordinate consistency
            if len(ground_truth) > 0:
                gt_pos = ground_truth[0]['position']
                print(f"  Ground truth absolute position: {gt_pos}")
                print(f"  Relative to satellite: {gt_pos - satellite_pos}")
                print(f"  Expected range: {np.linalg.norm(gt_pos - satellite_pos):.1f}")
                print(f"  Measured range: {measurements[0]['range']:.1f}")
    
    ospa_distance = calculate_ospa(ground_truth, extracted_state, 1000.0, 1.0)
    
    print(f"  Measurements: {len(measurements)}")
    print(f"  Tracks: {len(extracted_state)}")
    print(f"  OSPA: {ospa_distance:.2f}")
    
    if ospa_distance < 1000.0:  # Should be low with good tracking (increased threshold)
        print("  ✓ End-to-end pipeline working")
        return True
    else:
        print("  ✗ End-to-end pipeline not working well")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SIMPLE TRACKING COMPONENT TESTS")
    print("=" * 60)
    
    tests = [
        ("Perception Layer", test_perception_layer),
        ("Estimation Layer", test_estimation_layer), 
        ("OSPA Calculation", test_ospa_calculation),
        ("End-to-End Pipeline", test_end_to_end)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ {test_name} failed with error: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    if not all_passed:
        print("\nNext steps:")
        print("1. Fix failing components")
        print("2. Re-run tests")
        print("3. Only then restart training")

if __name__ == "__main__":
    main() 