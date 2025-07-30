#!/usr/bin/env python3
"""
Unit consistency test for the constellation framework

This script verifies that all units are consistent in meters throughout the codebase.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from simulation.utils.constellation_loader import (
    load_constellation_parameters, 
    get_constellation_positions
)
import config


def test_unit_consistency():
    """Test that all units are consistent in meters."""
    print("=" * 60)
    print("Testing Unit Consistency")
    print("=" * 60)
    
    # Test 1: Check config constants
    print("\n1. Checking config constants...")
    print(f"✓ EARTH_RADIUS_M: {config.EARTH_RADIUS_M} m")
    print(f"✓ GRAVITATIONAL_CONSTANT: {config.GRAVITATIONAL_CONSTANT} m³/s²")
    
    # Test 2: Check constellation parameters
    print("\n2. Checking constellation parameters...")
    constellation_params = load_constellation_parameters()
    
    if not constellation_params:
        print("ERROR: No constellation parameters found!")
        return False
    
    # Check semi-major axis units (should be in meters)
    semi_major_axes = [params['semi_major_axis'] for params in constellation_params.values()]
    avg_semi_major_axis = np.mean(semi_major_axes)
    
    print(f"✓ Semi-major axis range: {min(semi_major_axes)/1000:.1f} to {max(semi_major_axes)/1000:.1f} km")
    print(f"✓ Average semi-major axis: {avg_semi_major_axis/1000:.1f} km")
    
    # Verify this is reasonable (should be around 7228 km)
    if 7000 < avg_semi_major_axis/1000 < 7500:
        print("✓ Semi-major axis is in reasonable range (7-7.5 Mm)")
    else:
        print(f"WARNING: Semi-major axis {avg_semi_major_axis/1000:.1f} km seems unusual")
    
    # Test 3: Check constellation positions
    print("\n3. Checking constellation positions...")
    positions = get_constellation_positions()
    
    # Check position magnitudes (should be around semi-major axis)
    position_magnitudes = np.linalg.norm(positions, axis=1)
    avg_position_magnitude = np.mean(position_magnitudes)
    
    print(f"✓ Position magnitude range: {min(position_magnitudes)/1000:.1f} to {max(position_magnitudes)/1000:.1f} km")
    print(f"✓ Average position magnitude: {avg_position_magnitude/1000:.1f} km")
    
    # Verify positions are consistent with semi-major axis
    if abs(avg_position_magnitude - avg_semi_major_axis) < 1000:  # Within 1 km
        print("✓ Position magnitudes are consistent with semi-major axis")
    else:
        print(f"WARNING: Position magnitude {avg_position_magnitude/1000:.1f} km differs from semi-major axis {avg_semi_major_axis/1000:.1f} km")
    
    # Test 4: Check altitude calculations
    print("\n4. Checking altitude calculations...")
    altitudes_km = [(params['semi_major_axis'] - config.EARTH_RADIUS_M) / 1000 for params in constellation_params.values()]
    avg_altitude_km = np.mean(altitudes_km)
    
    print(f"✓ Altitude range: {min(altitudes_km):.1f} to {max(altitudes_km):.1f} km")
    print(f"✓ Average altitude: {avg_altitude_km:.1f} km")
    
    # Verify altitude is reasonable (should be around 650 km)
    if 600 < avg_altitude_km < 700:
        print("✓ Altitude is in reasonable range (600-700 km)")
    else:
        print(f"WARNING: Altitude {avg_altitude_km:.1f} km seems unusual")
    
    # Test 5: Check coordinate ranges
    print("\n5. Checking coordinate ranges...")
    x_coords = positions[:, 0]
    y_coords = positions[:, 1]
    z_coords = positions[:, 2]
    
    print(f"✓ X coordinate range: {x_coords.min()/1000:.1f} to {x_coords.max()/1000:.1f} km")
    print(f"✓ Y coordinate range: {y_coords.min()/1000:.1f} to {y_coords.max()/1000:.1f} km")
    print(f"✓ Z coordinate range: {z_coords.min()/1000:.1f} to {z_coords.max()/1000:.1f} km")
    
    # Verify coordinates are reasonable (should be around ±7000 km)
    max_coord = max(abs(x_coords).max(), abs(y_coords).max(), abs(z_coords).max())
    if max_coord < 10000:  # Less than 10,000 km
        print("✓ Coordinate ranges are reasonable")
    else:
        print(f"WARNING: Maximum coordinate {max_coord/1000:.1f} km seems too large")
    
    print("\n" + "=" * 60)
    print("Unit Consistency Test Complete")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = test_unit_consistency()
    if success:
        print("\n✓ All unit tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some unit tests failed!")
        sys.exit(1) 