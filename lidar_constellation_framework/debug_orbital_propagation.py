#!/usr/bin/env python3
"""
Debug script to understand the orbital propagation issue.
"""

import numpy as np
import sys
import os

# Add the data_generation directory to the path
sys.path.append('data_generation')
from preprocess_cpe_files_memory_efficient import simplified_orbital_propagation

def debug_orbital_propagation():
    """Debug the orbital propagation function."""
    
    print("🔍 Debugging Orbital Propagation")
    print("=" * 50)
    
    # Test with a sample orbital element set (from the object ID: 0.0776454_854.2_48.25_59.45)
    test_elements = {
        'SemiMajorAxis': 854200.0,  # 854.2 km in meters
        'Eccntr': 0.0776454,        # From the object ID
        'Incl': 48.25,              # From the object ID
    }
    
    print("📊 Test Orbital Elements:")
    for key, value in test_elements.items():
        print(f"  {key}: {value}")
    
    print(f"\n🔧 Expected Values:")
    print(f"  Semi-major axis: {test_elements['SemiMajorAxis']} m = {test_elements['SemiMajorAxis'] / 1000} km")
    print(f"  Earth radius: 6371 km = 6371000 m")
    print(f"  Expected altitude: {(test_elements['SemiMajorAxis'] / 1000) - 6371} km")
    
    # Test the propagation function
    time_deltas = np.array([0.0])  # Just the initial position
    print(f"\n🚀 Testing Propagation...")
    
    try:
        states = simplified_orbital_propagation(test_elements, time_deltas)
        
        if states:
            position, velocity = states[0]
            pos_magnitude = np.linalg.norm(position)
            vel_magnitude = np.linalg.norm(velocity)
            
            print(f"✅ Propagation Result:")
            print(f"  Position: {position}")
            print(f"  Position magnitude: {pos_magnitude:.1f} m")
            print(f"  Velocity: {velocity}")
            print(f"  Velocity magnitude: {vel_magnitude:.1f} m/s")
            
            # Calculate altitude
            earth_radius = 6371000  # m
            altitude = pos_magnitude - earth_radius
            
            print(f"\n📏 Analysis:")
            print(f"  Distance from Earth center: {pos_magnitude/1000:.1f} km")
            print(f"  Altitude above Earth surface: {altitude/1000:.1f} km")
            print(f"  Expected altitude: {(test_elements['SemiMajorAxis'] / 1000) - 6371:.1f} km")
            
            # Check if this matches what we see in the database
            print(f"\n🔍 Comparison with Database:")
            print(f"  Database shows: ~850 m magnitude")
            print(f"  Propagation gives: {pos_magnitude:.1f} m magnitude")
            print(f"  Ratio: {pos_magnitude / 850:.1f}")
            
            if abs(pos_magnitude - 850) < 10:
                print(f"  ✅ This matches the database! The issue is in the propagation.")
            else:
                print(f"  ❌ This doesn't match the database. There's another issue.")
                
        else:
            print("❌ No states returned from propagation")
            
    except Exception as e:
        print(f"❌ Error in propagation: {e}")
        import traceback
        traceback.print_exc()

def test_manual_calculation():
    """Test manual orbital calculation to understand the issue."""
    
    print(f"\n🧮 Manual Calculation Test")
    print("=" * 50)
    
    # Test with the same elements
    a_km = 854.2
    a_m = a_km * 1000  # 854,200 m
    
    print(f"  Semi-major axis: {a_km} km = {a_m} m")
    
    # For a circular orbit at t=0, the position should be at the semi-major axis
    # But the propagation is giving us ~850m, which suggests a unit issue
    
    # Let's check what happens if we don't convert units
    a_wrong = 854.2  # meters (wrong)
    print(f"  If treated as meters: {a_wrong} m")
    print(f"  Ratio: {a_m / a_wrong}")
    
    # The ratio is 1000, which suggests the issue is that the semi-major axis
    # is being treated as meters instead of kilometers
    
    # But wait, let's check the gravitational parameter
    mu_km = 398600.4418  # km³/s²
    mu_m = mu_km * 1e9   # m³/s²
    
    print(f"\n  Gravitational parameter:")
    print(f"    km³/s²: {mu_km}")
    print(f"    m³/s²: {mu_m}")
    
    # Calculate orbital period
    T_km = 2 * np.pi * np.sqrt(a_km**3 / mu_km)
    T_m = 2 * np.pi * np.sqrt(a_m**3 / mu_m)
    
    print(f"\n  Orbital period:")
    print(f"    Using km: {T_km:.1f} s = {T_km/3600:.1f} hours")
    print(f"    Using m: {T_m:.1f} s = {T_m/3600:.1f} hours")
    
    # These should be the same! Let's check:
    print(f"    Ratio: {T_m / T_km}")
    
    if abs(T_m / T_km - 1.0) < 0.01:
        print(f"    ✅ Periods match - unit conversion is correct")
    else:
        print(f"    ❌ Periods don't match - unit conversion issue")

if __name__ == "__main__":
    debug_orbital_propagation()
    test_manual_calculation() 