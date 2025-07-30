#!/usr/bin/env python3
"""
Simple test to verify orbital mechanics implementation.
"""

import numpy as np
import sys
import os

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from environment.estimation_layer import EstimationLayer

def test_simple_orbital():
    """Test orbital mechanics with simple circular orbit."""
    print("Simple Orbital Mechanics Test")
    print("=" * 40)
    
    # Create estimation layer
    estimation_layer = EstimationLayer()
    
    # Test circular orbit at 400km altitude
    # Initial position: [R_earth + 400km, 0, 0] = [6778137, 0, 0] meters
    # Initial velocity: Circular orbit velocity ~7660 m/s in Y direction
    initial_state = np.array([6778137.0, 0.0, 0.0, 0.0, 7660.0, 0.0])
    
    print(f"Initial position: {initial_state[:3]} meters")
    print(f"Initial velocity: {initial_state[3:]} m/s")
    print(f"Initial altitude: {np.linalg.norm(initial_state[:3]) - 6378137:.1f} km")
    
    # Use smaller time step for better accuracy
    dt = 10.0  # 10 seconds
    current_state = initial_state.copy()
    
    # Track positions and check altitude
    for i in range(60):  # 10 minutes total
        current_state = estimation_layer._orbital_dynamics(current_state, dt)
        
        if i % 6 == 0:  # Print every minute
            pos = current_state[:3]
            vel = current_state[3:]
            altitude = np.linalg.norm(pos) - 6378137
            velocity_mag = np.linalg.norm(vel)
            
            print(f"Time: {(i+1)*dt/60:.1f} min, Altitude: {altitude/1000:.1f} km, Velocity: {velocity_mag:.0f} m/s")
            
            # Check if altitude is reasonable (should stay around 400 km)
            if altitude > 1000000:  # More than 1000 km
                print(f"WARNING: Altitude too high! {altitude/1000:.1f} km")
                break
    
    print("\nTest Complete!")

if __name__ == "__main__":
    test_simple_orbital() 