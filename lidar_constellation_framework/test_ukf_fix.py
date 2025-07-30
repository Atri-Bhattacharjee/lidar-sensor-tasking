#!/usr/bin/env python3
"""
Test script to verify the UKF fix is working correctly.
"""

import sys
import os
import numpy as np

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

import config
from environment.estimation_layer import EstimationLayer, GaussianComponent, Track
from environment.perception_layer import PerceptionLayer


def test_ukf_update():
    """Test the UKF update functionality."""
    print("Testing UKF update functionality...")
    
    # Initialize estimation layer with perception layer
    perception_layer = PerceptionLayer()
    estimation_layer = EstimationLayer(perception_layer)
    
    # Create a test track with a single Gaussian component
    initial_mean = np.array([1000000, 500000, 200000, 7500, 0, 0]).reshape(6, 1)
    initial_covariance = np.eye(6) * 10000  # Large initial uncertainty
    initial_component = GaussianComponent(1.0, initial_mean, initial_covariance)
    
    track = Track(
        label=1,
        existence_probability=0.1,  # Low initial existence
        gmm=[initial_component]
    )
    
    print(f"Initial track state:")
    print(f"  Existence probability: {track.existence_probability:.3f}")
    print(f"  Mean position: {track.gmm[0].mean[:3].flatten()}")
    print(f"  Mean velocity: {track.gmm[0].mean[3:].flatten()}")
    print(f"  Covariance trace: {np.trace(track.gmm[0].covariance):.2e}")
    
    # Create a test measurement
    measurement = {
        'range': 1000000.0,
        'azimuth': 26.57,  # ~26.57 degrees
        'elevation': 11.31,  # ~11.31 degrees
        'range_rate': 7500.0,
        'satellite_id': 0,
        'timestamp': 0.0
    }
    
    print(f"\nTest measurement:")
    print(f"  Range: {measurement['range']:.1f} m")
    print(f"  Azimuth: {measurement['azimuth']:.2f} deg")
    print(f"  Elevation: {measurement['elevation']:.2f} deg")
    print(f"  Range rate: {measurement['range_rate']:.1f} m/s")
    
    # Update the track with the measurement
    updated_track = estimation_layer._update_track_with_measurement(track, measurement)
    
    print(f"\nUpdated track state:")
    print(f"  Existence probability: {updated_track.existence_probability:.3f}")
    print(f"  Mean position: {updated_track.gmm[0].mean[:3].flatten()}")
    print(f"  Mean velocity: {updated_track.gmm[0].mean[3:].flatten()}")
    print(f"  Covariance trace: {np.trace(updated_track.gmm[0].covariance):.2e}")
    
    # Check if the update was successful
    covariance_reduced = np.trace(updated_track.gmm[0].covariance) < np.trace(initial_covariance)
    existence_increased = updated_track.existence_probability > track.existence_probability
    
    print(f"\nUpdate results:")
    print(f"  Covariance reduced: {covariance_reduced}")
    print(f"  Existence increased: {existence_increased}")
    
    if covariance_reduced and existence_increased:
        print("✅ UKF update is working correctly!")
        return True
    else:
        print("❌ UKF update may have issues!")
        return False


def test_track_extraction():
    """Test track extraction functionality."""
    print("\nTesting track extraction functionality...")
    
    # Initialize estimation layer
    perception_layer = PerceptionLayer()
    estimation_layer = EstimationLayer(perception_layer)
    
    # Create a track with high existence probability
    initial_mean = np.array([1000000, 500000, 200000, 7500, 0, 0]).reshape(6, 1)
    initial_covariance = np.eye(6) * 1000  # Smaller uncertainty
    initial_component = GaussianComponent(1.0, initial_mean, initial_covariance)
    
    track = Track(
        label=1,
        existence_probability=0.06,  # Above extraction threshold
        gmm=[initial_component]
    )
    
    # Add track to estimation layer
    estimation_layer.tracks = [track]
    estimation_layer.track_max_existence[1] = 0.06  # Set max existence
    
    # Test extraction
    extracted_state = estimation_layer._extract_state()
    
    print(f"Track existence probability: {track.existence_probability:.3f}")
    print(f"Track max existence: {estimation_layer.track_max_existence[1]:.3f}")
    print(f"Number of extracted objects: {len(extracted_state)}")
    
    if len(extracted_state) > 0:
        print("✅ Track extraction is working!")
        return True
    else:
        print("❌ Track extraction may have issues!")
        return False


def test_full_filter_step():
    """Test a full filter step with measurements."""
    print("\nTesting full filter step...")
    
    # Initialize estimation layer
    perception_layer = PerceptionLayer()
    estimation_layer = EstimationLayer(perception_layer)
    
    # Create test measurements
    measurements = [
        {
            'range': 1000000.0,
            'azimuth': 26.57,
            'elevation': 11.31,
            'range_rate': 7500.0,
            'satellite_id': 0,
            'timestamp': 0.0
        },
        {
            'range': 1200000.0,
            'azimuth': 30.0,
            'elevation': 15.0,
            'range_rate': 8000.0,
            'satellite_id': 1,
            'timestamp': 0.0
        }
    ]
    
    # Run filter step
    extracted_state, unassociated = estimation_layer.step(measurements, [], dt=1.0)
    
    print(f"Number of measurements: {len(measurements)}")
    print(f"Number of extracted objects: {len(extracted_state)}")
    print(f"Number of unassociated measurements: {len(unassociated)}")
    
    if len(extracted_state) > 0:
        print("✅ Full filter step is working!")
        return True
    else:
        print("❌ Full filter step may have issues!")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("UKF Fix Verification Tests")
    print("=" * 60)
    
    # Run tests
    test1_passed = test_ukf_update()
    test2_passed = test_track_extraction()
    test3_passed = test_full_filter_step()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"UKF Update Test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Track Extraction Test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    print(f"Full Filter Step Test: {'✅ PASSED' if test3_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed and test3_passed:
        print("\n🎉 All tests passed! The UKF fix should resolve the zero detection rate issue.")
    else:
        print("\n⚠️  Some tests failed. There may still be issues to address.")


if __name__ == "__main__":
    main() 