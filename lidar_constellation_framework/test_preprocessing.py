"""
Test script for CPE file preprocessing functionality.

This script tests the preprocessing pipeline to ensure it works correctly
with the provided .cpe files.
"""

import os
import sys
import pickle
import numpy as np

# Add the data_generation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_generation'))

from preprocess_cpe_files import (
    load_and_combine_cpe_files,
    identify_unique_objects,
    create_ground_truth_database
)


def test_preprocessing_pipeline():
    """Test the complete preprocessing pipeline."""
    print("=" * 60)
    print("Testing CPE File Preprocessing Pipeline")
    print("=" * 60)
    
    # Configuration
    source_directory = "output"
    test_output_directory = "test_ground_truth_data"
    
    # Create test output directory
    os.makedirs(test_output_directory, exist_ok=True)
    
    try:
        # Check if source directory exists
        if not os.path.exists(source_directory):
            print(f"❌ Source directory '{source_directory}' not found!")
            print("Please ensure the 'output' directory contains *_cond.cpe files.")
            return False
        
        # Step 1: Test loading and combining CPE files
        print("\nStep 1: Testing CPE file loading...")
        try:
            combined_df = load_and_combine_cpe_files(source_directory)
            print(f"✅ Successfully loaded {len(combined_df)} total detections")
            print(f"   DataFrame shape: {combined_df.shape}")
            print(f"   Columns: {list(combined_df.columns)}")
        except Exception as e:
            print(f"❌ Failed to load CPE files: {e}")
            return False
        
        # Step 2: Test unique object identification
        print("\nStep 2: Testing unique object identification...")
        try:
            initial_states_df = identify_unique_objects(combined_df)
            print(f"✅ Successfully identified {len(initial_states_df)} unique objects")
            print(f"   Sample unique ID: {initial_states_df['UniqueID'].iloc[0]}")
        except Exception as e:
            print(f"❌ Failed to identify unique objects: {e}")
            return False
        
        # Step 3: Test ground truth database creation (with smaller time range for testing)
        print("\nStep 3: Testing ground truth database creation...")
        try:
            # Use a smaller time range for testing
            test_start_epoch = 2025.0
            test_end_epoch = 2025.1  # Just 36 days for testing
            test_time_delta = 60.0  # 1 minute steps for testing
            
            ground_truth_database = create_ground_truth_database(
                initial_states_df,
                test_start_epoch,
                test_end_epoch,
                test_time_delta
            )
            
            print(f"✅ Successfully created test ground truth database")
            print(f"   Number of timesteps: {len(ground_truth_database)}")
            
            # Test a few timesteps
            sample_timesteps = list(ground_truth_database.keys())[:5]
            for timestep in sample_timesteps:
                objects = ground_truth_database[timestep]
                print(f"   Timestep {timestep}: {len(objects)} objects")
                
                if objects:
                    sample_obj = objects[0]
                    print(f"     Sample object: {sample_obj['UniqueID']}")
                    print(f"     Position shape: {sample_obj['position'].shape}")
                    print(f"     State vector shape: {sample_obj['state_vector'].shape}")
                    break
            
        except Exception as e:
            print(f"❌ Failed to create ground truth database: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Step 4: Test saving and loading
        print("\nStep 4: Testing save and load functionality...")
        try:
            test_output_path = os.path.join(test_output_directory, "test_ground_truth_database.pkl")
            
            # Save
            with open(test_output_path, 'wb') as f:
                pickle.dump(ground_truth_database, f)
            
            # Load
            with open(test_output_path, 'rb') as f:
                loaded_database = pickle.load(f)
            
            print(f"✅ Successfully saved and loaded test database")
            print(f"   File size: {os.path.getsize(test_output_path) / 1024:.1f} KB")
            print(f"   Loaded timesteps: {len(loaded_database)}")
            
            # Verify data integrity
            assert len(ground_truth_database) == len(loaded_database), "Data integrity check failed"
            print(f"   Data integrity: ✅ Verified")
            
        except Exception as e:
            print(f"❌ Failed to save/load test database: {e}")
            return False
        
        print("\n" + "=" * 60)
        print("✅ All preprocessing tests passed!")
        print("=" * 60)
        
        # Provide instructions for full processing
        print("\nTo create the full ground truth database:")
        print("1. Run: python data_generation/preprocess_cpe_files.py")
        print("2. This will create: ground_truth_data/ground_truth_database.pkl")
        print("3. The simulation will automatically use this database")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_integration():
    """Test that the environment can load the ground truth database."""
    print("\n" + "=" * 60)
    print("Testing Environment Integration")
    print("=" * 60)
    
    try:
        # Add simulation directory to path
        sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))
        
        from environment.constellation_env import ConstellationEnv
        
        print("Creating constellation environment...")
        env = ConstellationEnv()
        
        print("✅ Environment created successfully")
        print(f"   Ground truth type: {type(env.ground_truth)}")
        
        if hasattr(env, 'ground_truth'):
            print(f"   Ground truth timesteps: {len(env.ground_truth)}")
            
            # Test accessing a timestep
            sample_timestep = min(env.ground_truth.keys())
            sample_objects = env.ground_truth[sample_timestep]
            print(f"   Sample timestep {sample_timestep}: {len(sample_objects)} objects")
            
            if sample_objects:
                sample_obj = sample_objects[0]
                print(f"   Sample object keys: {list(sample_obj.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run tests
    preprocessing_success = test_preprocessing_pipeline()
    environment_success = test_environment_integration()
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    print(f"Preprocessing Pipeline: {'✅ PASSED' if preprocessing_success else '❌ FAILED'}")
    print(f"Environment Integration: {'✅ PASSED' if environment_success else '❌ FAILED'}")
    
    if preprocessing_success and environment_success:
        print("\n🎉 All tests passed! The preprocessing system is ready to use.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1) 