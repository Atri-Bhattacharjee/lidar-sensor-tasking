# Quick test script to estimate preprocessing time
# Processes only a small subset to get timing estimates

import os
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Iterator, Tuple
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def define_cpe_columns() -> List[str]:
    """Define the expected columns for CPE files."""
    return [
        'CPE_No', 'Flux_Contrib', 'Target_Resid', 'Sc', 'Object_Mass', 'Object_Diam.',
        'Tr.Lat', 'Veloc.', 'Azimuth', 'Elevat.', 'Altitd.', 'RghtAsc', 'Declin.',
        'Srf_Vl', 'Srf_Ang', 'Epoch', 'Ballst_Limit', 'Conchoid_Diam', 
        'SemiMajorAxis', 'Eccntr', 'Incl.'
    ]

def test_file_processing(source_directory: str, max_files: int = 2, max_rows_per_file: int = 1000):
    """Test processing a small subset of files to estimate timing."""
    
    start_time = time.time()
    
    # Find _cond.cpe files
    cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    cpe_files = cpe_files[:max_files]  # Only process first few files
    
    logger.info(f"Testing with {len(cpe_files)} files: {cpe_files}")
    
    total_rows = 0
    unique_objects = set()
    
    for cpe_file in cpe_files:
        file_start = time.time()
        file_path = os.path.join(source_directory, cpe_file)
        
        try:
            # Read only first few rows
            df = pd.read_csv(
                file_path,
                sep=r'\s+',
                names=define_cpe_columns(),
                nrows=max_rows_per_file,
                engine='python',
                skiprows=1,
                on_bad_lines='skip'
            )
            
            file_rows = len(df)
            total_rows += file_rows
            
            # Count unique objects (simplified)
            for _, row in df.iterrows():
                try:
                    if pd.notna(row['SemiMajorAxis']) and pd.notna(row['Eccntr']) and pd.notna(row['Incl.']):
                        orbital_signature = f"{row['SemiMajorAxis']:.6f}_{row['Eccntr']:.6f}_{row['Incl.']:.6f}"
                        unique_objects.add(orbital_signature)
                except:
                    continue
            
            file_time = time.time() - file_start
            logger.info(f"File {cpe_file}: {file_rows} rows, {file_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing {cpe_file}: {e}")
            continue
    
    total_time = time.time() - start_time
    logger.info(f"Total test time: {total_time:.2f}s")
    logger.info(f"Total rows processed: {total_rows}")
    logger.info(f"Unique objects found: {len(unique_objects)}")
    
    return total_time, total_rows, len(unique_objects), len(cpe_files)

def estimate_full_processing_time(test_time: float, test_rows: int, test_files: int, 
                                test_unique_objects: int, source_directory: str):
    """Estimate full processing time based on test results."""
    
    # Count total files and estimate total rows
    all_cpe_files = [f for f in os.listdir(source_directory) if f.endswith('_cond.cpe')]
    total_files = len(all_cpe_files)
    
    logger.info(f"Total _cond.cpe files: {total_files}")
    logger.info(f"Test processed: {test_files} files")
    
    # Estimate total rows (assuming similar file sizes)
    estimated_total_rows = (test_rows / test_files) * total_files
    
    # Estimate unique objects (assuming similar distribution)
    estimated_unique_objects = (test_unique_objects / test_files) * total_files
    
    # Estimate file processing time
    file_processing_time = (test_time / test_files) * total_files
    
    # Estimate orbital propagation time (525,960 timesteps for 3 months)
    timesteps = 525960
    objects_per_second = test_unique_objects / test_time  # rough estimate
    propagation_time = estimated_unique_objects / objects_per_second * (timesteps / 1000)  # scaled
    
    total_estimated_time = file_processing_time + propagation_time
    
    logger.info("=" * 60)
    logger.info("TIME ESTIMATES:")
    logger.info("=" * 60)
    logger.info(f"Estimated total rows: {estimated_total_rows:,.0f}")
    logger.info(f"Estimated unique objects: {estimated_unique_objects:,.0f}")
    logger.info(f"File processing time: {file_processing_time/60:.1f} minutes")
    logger.info(f"Orbital propagation time: {propagation_time/60:.1f} minutes")
    logger.info(f"Total estimated time: {total_estimated_time/60:.1f} minutes")
    logger.info(f"Total estimated time: {total_estimated_time/3600:.1f} hours")

def main():
    """Run the speed test."""
    source_directory = os.path.join(os.path.dirname(__file__), "output")
    
    if not os.path.exists(source_directory):
        logger.error(f"Source directory does not exist: {source_directory}")
        return
    
    logger.info("Starting preprocessing speed test...")
    logger.info(f"Source directory: {source_directory}")
    
    # Run test with small subset
    test_time, test_rows, test_unique_objects, test_files = test_file_processing(
        source_directory, 
        max_files=2,  # Only test 2 files
        max_rows_per_file=1000  # Only test 1000 rows per file
    )
    
    # Estimate full processing time
    estimate_full_processing_time(test_time, test_rows, test_files, test_unique_objects, source_directory)

if __name__ == "__main__":
    main() 