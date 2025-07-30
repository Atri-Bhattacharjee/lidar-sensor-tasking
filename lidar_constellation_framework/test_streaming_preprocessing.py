#!/usr/bin/env python3
"""
Test script for streaming preprocessing.
"""

import os
import sys
import pickle
import numpy as np

# Add the data_generation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_generation'))

from preprocess_cpe_files_streaming import create_ground_truth_database_streaming

def test_streaming_preprocessing():
    """Test the streaming preprocessing pipeline."""
    
    print("="*60)
    print("TESTING STREAMING PREPROCESSING")
    print("="*60)
    
    # Configuration
    source_directory = os.path.join("output")
    output_directory = os.path.join("ground_truth_data")
    output_filename = "ground_truth_database_streaming_test.pkl"
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"❌ Source directory '{source_directory}' not found!")
        print("Please ensure the 'output' directory contains *_cond.cpe files.")
        return False
    
    # Check for CPE files
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    if not cpe_files:
        print(f"❌ No *_cond.cpe files found in {source_directory}")
        return False
    
    print(f"✅ Found {len(cpe_files)} CPE files")
    
    # Test with a smaller time range for faster testing
    START_EPOCH_YR = 2025.0
    END_EPOCH_YR = 2025.1  # Only 1 month for testing
    TIME_DELTA_SECONDS = 30.0  # 30-second timesteps for testing
    
    print(f"Test parameters:")
    print(f"  Start epoch: {START_EPOCH_YR}")
    print(f"  End epoch: {END_EPOCH_YR}")
    print(f"  Time delta: {TIME_DELTA_SECONDS} seconds")
    
    # Create output directory
    os.makedirs(output_directory, exist_ok=True)
    
    # Run streaming preprocessing
    output_path = os.path.join(output_directory, output_filename)
    
    print(f"\nRunning streaming preprocessing...")
    success = create_ground_truth_database_streaming(
        source_directory,
        START_EPOCH_YR,
        END_EPOCH_YR,
        TIME_DELTA_SECONDS,
        output_path
    )
    
    if not success:
        print("❌ Streaming preprocessing failed!")
        return False
    
    # Test loading the database
    print(f"\nTesting database loading...")
    try:
        with open(output_path, 'rb') as f:
            ground_truth = pickle.load(f)
        
        print(f"✅ Successfully loaded streaming database")
        print(f"   • Total timesteps: {len(ground_truth)}")
        
        # Calculate statistics
        object_counts = [len(objects) for objects in ground_truth.values()]
        total_objects = sum(object_counts)
        avg_objects = np.mean(object_counts)
        min_objects = min(object_counts)
        max_objects = max(object_counts)
        
        print(f"   • Total object states: {total_objects}")
        print(f"   • Average objects per timestep: {avg_objects:.1f}")
        print(f"   • Min objects per timestep: {min_objects}")
        print(f"   • Max objects per timestep: {max_objects}")
        
        # Compare with original database
        original_path = os.path.join("ground_truth_data", "ground_truth_database.pkl")
        if os.path.exists(original_path):
            with open(original_path, 'rb') as f:
                original_ground_truth = pickle.load(f)
            
            original_counts = [len(objects) for objects in original_ground_truth.values()]
            original_avg = np.mean(original_counts)
            
            print(f"\nComparison with original database:")
            print(f"   • Original avg objects per timestep: {original_avg:.1f}")
            print(f"   • Streaming avg objects per timestep: {avg_objects:.1f}")
            print(f"   • Improvement: {avg_objects/original_avg:.1f}x more objects")
        
        # Sample some objects
        print(f"\nSample objects from timestep 0:")
        if 0 in ground_truth and ground_truth[0]:
            sample_obj = ground_truth[0][0]
            print(f"   Sample object keys: {list(sample_obj.keys())}")
            if 'position' in sample_obj:
                pos = sample_obj['position']
                print(f"   Sample position: [{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}]")
                print(f"   Position magnitude: {np.linalg.norm(pos):.1f} m")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading streaming database: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_streaming_preprocessing()
    if success:
        print("\n✅ Streaming preprocessing test completed successfully!")
    else:
        print("\n❌ Streaming preprocessing test failed!") 