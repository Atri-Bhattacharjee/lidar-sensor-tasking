#!/usr/bin/env python3
"""
Comprehensive test script to verify the ground truth database quality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def load_ground_truth_database(filepath: str = "ground_truth_data/ground_truth_database.pkl") -> Dict[int, List[Dict[str, Any]]]:
    """Load the ground truth database."""
    try:
        with open(filepath, 'rb') as f:
            database = pickle.load(f)
        print(f"✅ Successfully loaded ground truth database from {filepath}")
        return database
    except Exception as e:
        print(f"❌ Failed to load ground truth database: {e}")
        return {}

def test_database_structure(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test the basic structure of the database."""
    print("\n" + "="*60)
    print("TESTING DATABASE STRUCTURE")
    print("="*60)
    
    if not database:
        print("❌ Database is empty")
        return False
    
    # Check timesteps
    timesteps = sorted(database.keys())
    print(f"✅ Number of timesteps: {len(timesteps)}")
    print(f"✅ Timestep range: {min(timesteps)} to {max(timesteps)}")
    
    # Check first few timesteps
    for t in timesteps[:3]:
        objects = database[t]
        print(f"✅ Timestep {t}: {len(objects)} objects")
        
        if objects:
            # Check object structure
            obj = objects[0]
            required_keys = ['id', 'position', 'velocity', 'diameter', 'mass', 'source']
            missing_keys = [key for key in required_keys if key not in obj]
            
            if missing_keys:
                print(f"❌ Missing keys in object: {missing_keys}")
                return False
            else:
                print(f"✅ Object structure is correct")
    
    return True

def test_position_accuracy(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test that object positions are physically reasonable."""
    print("\n" + "="*60)
    print("TESTING POSITION ACCURACY")
    print("="*60)
    
    earth_radius = 6371000.0  # m
    leo_min_altitude = 200000.0  # m
    leo_max_altitude = 2000000.0  # m
    
    all_positions = []
    all_altitudes = []
    
    # Sample a few timesteps
    timesteps = sorted(database.keys())
    sample_timesteps = timesteps[::max(1, len(timesteps)//10)]  # Sample every 10th timestep
    
    for t in sample_timesteps:
        objects = database[t]
        for obj in objects:
            position = np.array(obj['position'])
            distance_from_earth_center = np.linalg.norm(position)
            altitude = distance_from_earth_center - earth_radius
            
            all_positions.append(position)
            all_altitudes.append(altitude)
    
    if not all_positions:
        print("❌ No positions found")
        return False
    
    # Convert to numpy arrays
    all_positions = np.array(all_positions)
    all_altitudes = np.array(all_altitudes)
    
    # Check altitude statistics
    print(f"✅ Altitude statistics:")
            print(f"   Min altitude: {np.min(all_altitudes)/1000:.1f} km")
        print(f"   Max altitude: {np.max(all_altitudes)/1000:.1f} km")
        print(f"   Mean altitude: {np.mean(all_altitudes)/1000:.1f} km")
        print(f"   Median altitude: {np.median(all_altitudes)/1000:.1f} km")
    
    # Check if altitudes are reasonable for LEO
    reasonable_altitudes = (all_altitudes >= leo_min_altitude) & (all_altitudes <= leo_max_altitude)
    reasonable_percentage = np.mean(reasonable_altitudes) * 100
    
    print(f"✅ {reasonable_percentage:.1f}% of objects have reasonable LEO altitudes")
    
    if reasonable_percentage < 90:
        print(f"⚠️  Warning: Only {reasonable_percentage:.1f}% of objects have reasonable altitudes")
        return False
    
    # Check position magnitudes
    position_magnitudes = np.linalg.norm(all_positions, axis=1)
    print(f"✅ Position magnitude statistics:")
            print(f"   Min distance from Earth center: {np.min(position_magnitudes)/1000:.1f} km")
        print(f"   Max distance from Earth center: {np.max(position_magnitudes)/1000:.1f} km")
        print(f"   Mean distance from Earth center: {np.mean(position_magnitudes)/1000:.1f} km")
    
    return True

def test_velocity_accuracy(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test that object velocities are physically reasonable."""
    print("\n" + "="*60)
    print("TESTING VELOCITY ACCURACY")
    print("="*60)
    
    all_velocities = []
    
    # Sample a few timesteps
    timesteps = sorted(database.keys())
    sample_timesteps = timesteps[::max(1, len(timesteps)//10)]
    
    for t in sample_timesteps:
        objects = database[t]
        for obj in objects:
            velocity = np.array(obj['velocity'])
            all_velocities.append(velocity)
    
    if not all_velocities:
        print("❌ No velocities found")
        return False
    
    all_velocities = np.array(all_velocities)
    velocity_magnitudes = np.linalg.norm(all_velocities, axis=1)
    
    # LEO orbital velocities should be around 7-8 km/s
    leo_min_velocity = 6000.0  # m/s
    leo_max_velocity = 9000.0  # m/s
    
    print(f"✅ Velocity magnitude statistics:")
            print(f"   Min velocity: {np.min(velocity_magnitudes)/1000:.2f} km/s")
        print(f"   Max velocity: {np.max(velocity_magnitudes)/1000:.2f} km/s")
        print(f"   Mean velocity: {np.mean(velocity_magnitudes)/1000:.2f} km/s")
        print(f"   Median velocity: {np.median(velocity_magnitudes)/1000:.2f} km/s")
    
    # Check if velocities are reasonable for LEO
    reasonable_velocities = (velocity_magnitudes >= leo_min_velocity) & (velocity_magnitudes <= leo_max_velocity)
    reasonable_percentage = np.mean(reasonable_velocities) * 100
    
    print(f"✅ {reasonable_percentage:.1f}% of objects have reasonable LEO velocities")
    
    if reasonable_percentage < 90:
        print(f"⚠️  Warning: Only {reasonable_percentage:.1f}% of objects have reasonable velocities")
        return False
    
    return True

def test_object_properties(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test that object properties are reasonable."""
    print("\n" + "="*60)
    print("TESTING OBJECT PROPERTIES")
    print("="*60)
    
    all_diameters = []
    all_masses = []
    unique_ids = set()
    
    # Sample a few timesteps
    timesteps = sorted(database.keys())
    sample_timesteps = timesteps[::max(1, len(timesteps)//10)]
    
    for t in sample_timesteps:
        objects = database[t]
        for obj in objects:
            all_diameters.append(obj['diameter'])
            all_masses.append(obj['mass'])
            unique_ids.add(obj['id'])
    
    if not all_diameters:
        print("❌ No object properties found")
        return False
    
    all_diameters = np.array(all_diameters)
    all_masses = np.array(all_masses)
    
    print(f"✅ Object statistics:")
    print(f"   Number of unique objects: {len(unique_ids)}")
    print(f"   Diameter range: {np.min(all_diameters):.3f} to {np.max(all_diameters):.3f} m")
    print(f"   Mass range: {np.min(all_masses):.1f} to {np.max(all_masses):.1f} kg")
    print(f"   Mean diameter: {np.mean(all_diameters):.3f} m")
    print(f"   Mean mass: {np.mean(all_masses):.1f} kg")
    
    # Check for reasonable debris sizes
    reasonable_diameters = (all_diameters > 0.01) & (all_diameters < 10.0)  # 1cm to 10m
    reasonable_masses = (all_masses > 0.001) & (all_masses < 10000.0)  # 1g to 10 tons
    
    diameter_percentage = np.mean(reasonable_diameters) * 100
    mass_percentage = np.mean(reasonable_masses) * 100
    
    print(f"✅ {diameter_percentage:.1f}% of objects have reasonable diameters")
    print(f"✅ {mass_percentage:.1f}% of objects have reasonable masses")
    
    return True

def test_temporal_consistency(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test that objects move consistently over time."""
    print("\n" + "="*60)
    print("TESTING TEMPORAL CONSISTENCY")
    print("="*60)
    
    # Track a few objects over time
    timesteps = sorted(database.keys())
    if len(timesteps) < 2:
        print("❌ Not enough timesteps for temporal consistency test")
        return False
    
    # Get objects from first timestep
    first_objects = database[timesteps[0]]
    if not first_objects:
        print("❌ No objects in first timestep")
        return False
    
    # Track first few objects
    num_objects_to_track = min(3, len(first_objects))
    tracked_positions = {obj['id']: [] for obj in first_objects[:num_objects_to_track]}
    
    # Sample timesteps
    sample_timesteps = timesteps[::max(1, len(timesteps)//20)]
    
    for t in sample_timesteps:
        objects = database[t]
        object_dict = {obj['id']: obj for obj in objects}
        
        for obj_id in tracked_positions.keys():
            if obj_id in object_dict:
                tracked_positions[obj_id].append(object_dict[obj_id]['position'])
    
    # Check movement consistency
    for obj_id, positions in tracked_positions.items():
        if len(positions) < 2:
            continue
            
        positions = np.array(positions)
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        
        print(f"✅ Object {obj_id}:")
        print(f"   Tracked for {len(positions)} timesteps")
        print(f"   Average movement per timestep: {np.mean(distances)/1000:.2f} km")
        print(f"   Max movement per timestep: {np.max(distances)/1000:.2f} km")
        
        # Check for reasonable movement (should be consistent)
        if np.std(distances) > np.mean(distances):
            print(f"⚠️  Warning: Inconsistent movement for object {obj_id}")
    
    return True

def test_coordinate_system(database: Dict[int, List[Dict[str, Any]]]) -> bool:
    """Test that the coordinate system is consistent."""
    print("\n" + "="*60)
    print("TESTING COORDINATE SYSTEM")
    print("="*60)
    
    # Sample positions from different timesteps
    all_positions = []
    timesteps = sorted(database.keys())
    sample_timesteps = timesteps[::max(1, len(timesteps)//10)]
    
    for t in sample_timesteps:
        objects = database[t]
        for obj in objects:
            all_positions.append(obj['position'])
    
    if not all_positions:
        print("❌ No positions found")
        return False
    
    all_positions = np.array(all_positions)
    
    # Check coordinate ranges
    print(f"✅ Coordinate ranges:")
            print(f"   X range: {np.min(all_positions[:, 0])/1000:.1f} to {np.max(all_positions[:, 0])/1000:.1f} km")
        print(f"   Y range: {np.min(all_positions[:, 1])/1000:.1f} to {np.max(all_positions[:, 1])/1000:.1f} km")
        print(f"   Z range: {np.min(all_positions[:, 2])/1000:.1f} to {np.max(all_positions[:, 2])/1000:.1f} km")
    
    # Check that positions are roughly spherical around Earth center
    distances = np.linalg.norm(all_positions, axis=1)
    earth_radius = 6371000.0  # m
    
    # Calculate spherical coordinates
    altitudes = distances - earth_radius
    latitudes = np.arcsin(all_positions[:, 2] / distances) * 180 / np.pi
    longitudes = np.arctan2(all_positions[:, 1], all_positions[:, 0]) * 180 / np.pi
    
    print(f"✅ Spherical coordinate statistics:")
    print(f"   Altitude range: {np.min(altitudes)/1000:.1f} to {np.max(altitudes)/1000:.1f} km")
    print(f"   Latitude range: {np.min(latitudes):.1f} to {np.max(latitudes):.1f} degrees")
    print(f"   Longitude range: {np.min(longitudes):.1f} to {np.max(longitudes):.1f} degrees")
    
    return True

def main():
    """Run all ground truth quality tests."""
    print("="*80)
    print("GROUND TRUTH DATABASE QUALITY TEST")
    print("="*80)
    
    # Load database
    database = load_ground_truth_database()
    if not database:
        return False
    
    # Run all tests
    tests = [
        ("Database Structure", test_database_structure),
        ("Position Accuracy", test_position_accuracy),
        ("Velocity Accuracy", test_velocity_accuracy),
        ("Object Properties", test_object_properties),
        ("Temporal Consistency", test_temporal_consistency),
        ("Coordinate System", test_coordinate_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func(database)
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_tests = sum(1 for _, result in results if result)
    total_tests = len(results)
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! Ground truth database is high quality.")
        return True
    else:
        print("⚠️  Some tests failed. Please review the warnings above.")
        return False

if __name__ == "__main__":
    main() 