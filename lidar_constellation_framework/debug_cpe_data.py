#!/usr/bin/env python3
"""
Debug script to examine the raw CPE data and understand the unit issue.
"""

import pandas as pd
import numpy as np
import glob
import os

def examine_cpe_file(filepath: str):
    """Examine a single CPE file to understand the data structure."""
    print(f"\nExamining file: {os.path.basename(filepath)}")
    print("="*60)
    
    # Define columns
    columns = [
        'CPE_No', 'Flux_Contrib', 'Target_Resid', 'Sc', 'Object_Mass', 'Object_Diam',
        'Tr_Lat', 'Veloc', 'Azimuth', 'Elevat', 'Altitd', 'RghtAsc', 'Declin',
        'Srf_Vl', 'Srf_Ang', 'Epoch', 'Ballst_Limit', 'Conchoid_Diam',
        'SemiMajorAxis', 'Eccntr', 'Incl', 'Unknown_Col'
    ]
    
    try:
        # Read first few rows
        df = pd.read_csv(filepath, sep=r'\s+', names=columns, comment='#', skiprows=1, nrows=10)
        
        print("First 5 rows of data:")
        print(df[['Object_Diam', 'SemiMajorAxis', 'Eccntr', 'Incl', 'Altitd', 'Epoch']].head())
        
        print(f"\nData statistics:")
        print(f"  SemiMajorAxis range: {df['SemiMajorAxis'].min():.3f} to {df['SemiMajorAxis'].max():.3f}")
        print(f"  Altitd range: {df['Altitd'].min():.3f} to {df['Altitd'].max():.3f}")
        print(f"  Object_Diam range: {df['Object_Diam'].min():.6f} to {df['Object_Diam'].max():.6f}")
        print(f"  Eccntr range: {df['Eccntr'].min():.6f} to {df['Eccntr'].max():.6f}")
        print(f"  Incl range: {df['Incl'].min():.2f} to {df['Incl'].max():.2f}")
        
        # Check if Altitd looks like it could be the actual semi-major axis
        print(f"\nAnalysis:")
        print(f"  Altitd values look like altitudes in km: {df['Altitd'].iloc[0]:.1f} km")
        print(f"  SemiMajorAxis values are very small: {df['SemiMajorAxis'].iloc[0]:.3f}")
        
        # Calculate what the semi-major axis should be
        earth_radius = 6371.0  # km
        altitude = df['Altitd'].iloc[0]
        expected_semi_major_axis = earth_radius + altitude
        print(f"  Expected semi-major axis for altitude {altitude:.1f} km: {expected_semi_major_axis:.1f} km")
        
        return df
        
    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def main():
    """Main function to examine CPE files."""
    print("="*80)
    print("CPE DATA ANALYSIS")
    print("="*80)
    
    # Find CPE files
    cpe_files = glob.glob("output/*_cond.cpe")
    
    if not cpe_files:
        print("No CPE files found in output directory")
        return
    
    print(f"Found {len(cpe_files)} CPE files")
    
    # Examine first few files
    for filepath in cpe_files[:3]:
        df = examine_cpe_file(filepath)
        if df is not None:
            break
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("The issue is clear: the 'SemiMajorAxis' column contains very small values")
    print("that are NOT actual semi-major axes. The 'Altitd' column contains")
    print("reasonable altitude values that should be used to calculate the")
    print("semi-major axis.")
    print("\nFIX: Use Altitd + Earth_radius as the semi-major axis instead of")
    print("the SemiMajorAxis column directly.")

if __name__ == "__main__":
    main() 