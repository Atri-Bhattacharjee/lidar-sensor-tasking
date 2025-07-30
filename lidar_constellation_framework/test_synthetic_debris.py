#!/usr/bin/env python3
"""
Test script for synthetic debris generation

This script tests the synthetic debris generator to ensure it creates
the correct number of debris objects with proper orbital parameters.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from simulation.utils.synthetic_debris_generator import (
    generate_synthetic_debris_objects,
    create_synthetic_ground_truth_database,
    save_synthetic_ground_truth
)
from simulation.utils.constellation_loader import load_constellation_parameters


def test_synthetic_debris_generation():
    """Test the synthetic debris generation functionality."""
    print("=" * 60)
    print("Testing Synthetic Debris Generation")
    print("=" * 60)
    
    # Test 1: Load constellation parameters
    print("\n1. Loading constellation parameters...")
    constellation_params = load_constellation_parameters()
    print(f"✓ Loaded {len(constellation_params)} satellites")
    
    # Test 2: Generate synthetic debris objects
    print("\n2. Generating synthetic debris objects...")
    debris_objects = generate_synthetic_debris_objects()
    print(f"✓ Generated {len(debris_objects)} debris objects")
    
    # Verify we have 12 debris per satellite
    expected_debris = len(constellation_params) * 12
    if len(debris_objects) == expected_debris:
        print(f"✓ Correct number of debris objects: {len(debris_objects)}")
    else:
        print(f"✗ Expected {expected_debris} debris objects, got {len(debris_objects)}")
        return False
    
    # Test 3: Check debris orbital parameters
    print("\n3. Checking debris orbital parameters...")
    
    # Check semi-major axis variations
    semi_major_axes = [debris['semi_major_axis'] for debris in debris_objects]
    base_semi_major = constellation_params['P1-S1']['semi_major_axis']
    variations = [abs(axis - base_semi_major) for axis in semi_major_axes]
    
    max_variation = max(variations) / 1000  # Convert to km
    print(f"✓ Maximum semi-major axis variation: {max_variation:.1f} km")
    
    if max_variation <= 500:
        print("✓ Semi-major axis variations are within ±500 km limit")
    else:
        print(f"✗ Semi-major axis variation {max_variation:.1f} km exceeds 500 km limit")
        return False
    
    # Check RAAN variations
    raan_variations = []
    for debris in debris_objects:
        parent_sat = debris['parent_satellite']
        parent_raan = constellation_params[parent_sat]['raan']
        debris_raan = debris['raan']
        variation = abs(debris_raan - parent_raan)
        # Handle wraparound
        variation = min(variation, 360 - variation)
        raan_variations.append(variation)
    
    max_raan_variation = max(raan_variations)
    print(f"✓ Maximum RAAN variation: {max_raan_variation:.1f} degrees")
    
    if max_raan_variation <= 10:
        print("✓ RAAN variations are within ±10 degrees limit")
    else:
        print(f"✗ RAAN variation {max_raan_variation:.1f} degrees exceeds 10 degrees limit")
        return False
    
    # Check mean anomaly variations
    mean_anomaly_variations = []
    for debris in debris_objects:
        parent_sat = debris['parent_satellite']
        parent_ma = constellation_params[parent_sat]['mean_anomaly']
        debris_ma = debris['mean_anomaly']
        variation = abs(debris_ma - parent_ma)
        # Handle wraparound
        variation = min(variation, 360 - variation)
        mean_anomaly_variations.append(variation)
    
    max_ma_variation = max(mean_anomaly_variations)
    print(f"✓ Maximum mean anomaly variation: {max_ma_variation:.1f} degrees")
    
    if max_ma_variation <= 10:
        print("✓ Mean anomaly variations are within ±10 degrees limit")
    else:
        print(f"✗ Mean anomaly variation {max_ma_variation:.1f} degrees exceeds 10 degrees limit")
        return False
    
    # Test 4: Check physical properties
    print("\n4. Checking physical properties...")
    diameters = [debris['diameter'] for debris in debris_objects]
    masses = [debris['mass'] for debris in debris_objects]
    
    print(f"✓ Diameter range: {min(diameters):.2f} - {max(diameters):.2f} m")
    print(f"✓ Mass range: {min(masses):.1f} - {max(masses):.1f} kg")
    
    # Test 5: Check parent satellite distribution
    print("\n5. Checking parent satellite distribution...")
    parent_counts = {}
    for debris in debris_objects:
        parent = debris['parent_satellite']
        parent_counts[parent] = parent_counts.get(parent, 0) + 1
    
    expected_per_satellite = 12
    for satellite, count in parent_counts.items():
        if count != expected_per_satellite:
            print(f"✗ Satellite {satellite} has {count} debris, expected {expected_per_satellite}")
            return False
    
    print(f"✓ All satellites have exactly {expected_per_satellite} debris objects")
    
    print("\n" + "=" * 60)
    print("✓ All synthetic debris generation tests passed!")
    print("=" * 60)
    return True


def test_ground_truth_creation():
    """Test the ground truth database creation."""
    print("\n" + "=" * 60)
    print("Testing Ground Truth Database Creation")
    print("=" * 60)
    
    try:
        # Create ground truth database
        print("\nCreating synthetic ground truth database...")
        ground_truth = create_synthetic_ground_truth_database()
        
        # Check database structure
        print(f"\n✓ Ground truth database created with {len(ground_truth)} timesteps")
        
        # Check first timestep
        first_timestep = ground_truth[0]
        print(f"✓ First timestep contains {len(first_timestep)} debris objects")
        
        # Check last timestep
        last_timestep = ground_truth[max(ground_truth.keys())]
        print(f"✓ Last timestep contains {len(last_timestep)} debris objects")
        
        # Check debris state structure
        if first_timestep:
            sample_debris = first_timestep[0]
            required_keys = ['debris_id', 'diameter', 'mass', 'state_vector', 'position', 'velocity', 'parent_satellite']
            
            for key in required_keys:
                if key not in sample_debris:
                    print(f"✗ Missing key '{key}' in debris state")
                    return False
            
            print("✓ Debris state structure is correct")
            
            # Check state vector dimensions
            state_vector = sample_debris['state_vector']
            if len(state_vector) == 6:  # [Px, Py, Pz, Vx, Vy, Vz]
                print("✓ State vector has correct dimensions (6)")
            else:
                print(f"✗ State vector has wrong dimensions: {len(state_vector)}")
                return False
        
        # Save the database
        print("\nSaving synthetic ground truth database...")
        save_synthetic_ground_truth(ground_truth)
        
        print("\n" + "=" * 60)
        print("✓ Ground truth database creation completed successfully!")
        print("=" * 60)
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating ground truth database: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        # Run debris generation tests
        success1 = test_synthetic_debris_generation()
        
        if success1:
            # Run ground truth creation tests
            success2 = test_ground_truth_creation()
            
            if success2:
                print("\n" + "=" * 60)
                print("✓ All tests completed successfully!")
                print("✓ Synthetic debris and ground truth database are ready for use!")
                print("=" * 60)
            else:
                print("\n✗ Ground truth creation tests failed!")
                sys.exit(1)
        else:
            print("\n✗ Synthetic debris generation tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 