#!/usr/bin/env python3
"""
Test script for constellation loader utility

This script tests the constellation loader to ensure it correctly reads
the orbital parameters from the .dia files and provides consistent data.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from simulation.utils.constellation_loader import (
    load_constellation_parameters, 
    get_constellation_positions, 
    get_constellation_summary
)


def test_constellation_loader():
    """Test the constellation loader functionality."""
    print("=" * 60)
    print("Testing Constellation Loader")
    print("=" * 60)
    
    # Test 1: Load constellation parameters
    print("\n1. Loading constellation parameters...")
    constellation_params = load_constellation_parameters()
    
    if not constellation_params:
        print("ERROR: No constellation parameters found!")
        return False
    
    print(f"✓ Loaded parameters for {len(constellation_params)} satellites")
    
    # Test 2: Check parameter consistency
    print("\n2. Checking parameter consistency...")
    
    # Check that all satellites have the same base parameters
    semi_major_axes = [params['semi_major_axis'] for params in constellation_params.values()]
    eccentricities = [params['eccentricity'] for params in constellation_params.values()]
    inclinations = [params['inclination'] for params in constellation_params.values()]
    
    if len(set(semi_major_axes)) != 1:
        print(f"WARNING: Inconsistent semi-major axes: {set(semi_major_axes)}")
    else:
        print(f"✓ All satellites have semi-major axis: {semi_major_axes[0]/1000:.1f} km")
    
    if len(set(eccentricities)) != 1:
        print(f"WARNING: Inconsistent eccentricities: {set(eccentricities)}")
    else:
        print(f"✓ All satellites have eccentricity: {eccentricities[0]}")
    
    if len(set(inclinations)) != 1:
        print(f"WARNING: Inconsistent inclinations: {set(inclinations)}")
    else:
        print(f"✓ All satellites have inclination: {inclinations[0]:.1f}°")
    
    # Test 3: Check RAAN distribution
    print("\n3. Checking RAAN distribution...")
    raans = [params['raan'] for params in constellation_params.values()]
    unique_raans = sorted(set(raans))
    print(f"✓ Found {len(unique_raans)} unique RAAN values: {unique_raans}")
    
    # Test 4: Check mean anomaly distribution
    print("\n4. Checking mean anomaly distribution...")
    mean_anomalies = [params['mean_anomaly'] for params in constellation_params.values()]
    unique_mean_anomalies = sorted(set(mean_anomalies))
    print(f"✓ Found {len(unique_mean_anomalies)} unique mean anomaly values: {unique_mean_anomalies}")
    
    # Check that mean anomalies are properly distributed (should be ~45° apart within each plane)
    print("✓ Mean anomaly distribution per plane:")
    for plane in range(1, 6):  # P1 to P5
        plane_anomalies = [params['mean_anomaly'] for sat_id, params in constellation_params.items() if sat_id.startswith(f'P{plane}')]
        print(f"  P{plane}: {sorted(plane_anomalies)}")
    
    # Test 5: Get constellation positions
    print("\n5. Getting constellation positions...")
    positions = get_constellation_positions()
    
    if len(positions) != len(constellation_params):
        print(f"ERROR: Position count ({len(positions)}) doesn't match parameter count ({len(constellation_params)})")
        return False
    
    print(f"✓ Generated positions for {len(positions)} satellites")
    
    # Check position ranges
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]
    
    print(f"✓ Position ranges:")
    print(f"  X: {x_coords.min()/1000:.1f} to {x_coords.max()/1000:.1f} km")
    print(f"  Y: {y_coords.min()/1000:.1f} to {y_coords.max()/1000:.1f} km")
    print(f"  Z: {z_coords.min()/1000:.1f} to {z_coords.max()/1000:.1f} km")
    
    # Test 6: Get constellation summary
    print("\n6. Getting constellation summary...")
    summary = get_constellation_summary()
    
    if 'error' in summary:
        print(f"ERROR: {summary['error']}")
        return False
    
    print(f"✓ Constellation Summary:")
    print(f"  Total satellites: {summary['total_satellites']}")
    print(f"  Number of planes: {summary['num_planes']}")
    print(f"  Satellites per plane: {summary['satellites_per_plane']}")
    print(f"  Inclination range: {summary['inclination_range']}")
    print(f"  Altitude range: {summary['altitude_range_km']}")
    
    # Test 7: Verify expected structure
    print("\n7. Verifying expected structure...")
    expected_planes = 5
    expected_sats_per_plane = 8
    expected_total = expected_planes * expected_sats_per_plane
    
    if summary['num_planes'] != expected_planes:
        print(f"WARNING: Expected {expected_planes} planes, got {summary['num_planes']}")
    else:
        print(f"✓ Correct number of planes: {expected_planes}")
    
    if summary['satellites_per_plane'] != expected_sats_per_plane:
        print(f"WARNING: Expected {expected_sats_per_plane} satellites per plane, got {summary['satellites_per_plane']}")
    else:
        print(f"✓ Correct satellites per plane: {expected_sats_per_plane}")
    
    if summary['total_satellites'] != expected_total:
        print(f"WARNING: Expected {expected_total} total satellites, got {summary['total_satellites']}")
    else:
        print(f"✓ Correct total satellites: {expected_total}")
    
    # Test 8: Check satellite IDs
    print("\n8. Checking satellite IDs...")
    satellite_ids = summary['satellite_ids']
    
    # Check that we have P1-S1 through P5-S8
    expected_ids = []
    for plane in range(1, 6):  # P1 to P5
        for sat in range(1, 9):  # S1 to S8
            expected_ids.append(f"P{plane}-S{sat}")
    
    missing_ids = set(expected_ids) - set(satellite_ids)
    extra_ids = set(satellite_ids) - set(expected_ids)
    
    if missing_ids:
        print(f"WARNING: Missing satellite IDs: {missing_ids}")
    else:
        print("✓ All expected satellite IDs present")
    
    if extra_ids:
        print(f"WARNING: Extra satellite IDs: {extra_ids}")
    else:
        print("✓ No extra satellite IDs")
    
    print("\n" + "=" * 60)
    print("Constellation Loader Test Complete")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_constellation_loader()
    if success:
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1) 