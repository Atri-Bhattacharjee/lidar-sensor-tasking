#!/usr/bin/env python3
"""
Test script to verify orbital dynamics implementation in the estimation layer.
"""

import numpy as np
import sys
import os

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from environment.estimation_layer import EstimationLayer

def test_orbital_dynamics():
    """Test the orbital dynamics implementation."""
    print("Testing Orbital Dynamics Implementation")
    print("=" * 50)
    
    # Create estimation layer
    estimation_layer = EstimationLayer()
    
    # Test case 1: Low Earth Orbit (LEO) satellite
    print("\nTest Case 1: Low Earth Orbit (LEO)")
    print("-" * 30)
    
    # Initial state: LEO at 400km altitude
    # Position: [R_earth + 400km, 0, 0] = [6778137, 0, 0] meters
    # Velocity: Circular orbit velocity ~7660 m/s in Y direction
    initial_state = np.array([6778137.0, 0.0, 0.0, 0.0, 7660.0, 0.0])
    
    print(f"Initial position: {initial_state[:3]} meters")
    print(f"Initial velocity: {initial_state[3:]} m/s")
    
    # Propagate for 1 hour (3600 seconds) in 60-second steps
    dt = 60.0  # 60 seconds
    current_state = initial_state.copy()
    
    positions = []
    velocities = []
    times = []
    
    for i in range(60):  # 60 steps = 1 hour
        current_state = estimation_layer._orbital_dynamics(current_state, dt)
        positions.append(current_state[:3].copy())
        velocities.append(current_state[3:].copy())
        times.append((i + 1) * dt)
        
        if i % 10 == 0:  # Print every 10 minutes
            print(f"Time: {times[-1]/60:.1f} min, Position: {current_state[:3]}, Velocity: {current_state[3:]}")
    
    # Check orbital period (should be roughly 90 minutes for LEO)
    # Calculate distance from initial position
    final_pos = np.array(positions[-1])
    initial_pos = np.array(positions[0])
    distance_traveled = np.linalg.norm(final_pos - initial_pos)
    
    print(f"\nDistance traveled in 1 hour: {distance_traveled/1000:.1f} km")
    print(f"Final altitude: {np.linalg.norm(final_pos) - 6378137:.1f} km")
    
    # Test case 2: Higher orbit (GEO-like)
    print("\nTest Case 2: Higher Orbit")
    print("-" * 30)
    
    # Initial state: Higher orbit at 35786 km altitude (GEO)
    # Position: [R_earth + 35786km, 0, 0] = [42164123, 0, 0] meters
    # Velocity: Circular orbit velocity ~3074 m/s in Y direction
    initial_state_geo = np.array([42164123.0, 0.0, 0.0, 0.0, 3074.0, 0.0])
    
    print(f"Initial position: {initial_state_geo[:3]} meters")
    print(f"Initial velocity: {initial_state_geo[3:]} m/s")
    
    # Propagate for 6 hours in 60-second steps
    current_state_geo = initial_state_geo.copy()
    
    for i in range(360):  # 360 steps = 6 hours
        current_state_geo = estimation_layer._orbital_dynamics(current_state_geo, dt)
        
        if i % 60 == 0:  # Print every hour
            print(f"Time: {(i + 1) * dt / 3600:.0f} hours, Position: {current_state_geo[:3]}, Velocity: {current_state_geo[3:]}")
    
    # Test case 3: Energy conservation
    print("\nTest Case 3: Energy Conservation")
    print("-" * 30)
    
    # Check if total energy (kinetic + potential) is conserved
    mu_earth = 3.986004418e14  # m³/s²
    
    for i, (pos, vel, t) in enumerate(zip(positions, velocities, times)):
        if i % 10 == 0:  # Check every 10 minutes
            r = np.linalg.norm(pos)
            v = np.linalg.norm(vel)
            
            # Total energy: E = 1/2 * v² - μ/r
            kinetic_energy = 0.5 * v**2
            potential_energy = -mu_earth / r
            total_energy = kinetic_energy + potential_energy
            
            print(f"Time: {t/60:.1f} min, Energy: {total_energy/1e6:.3f} MJ/kg")
    
    print("\nOrbital Dynamics Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_orbital_dynamics() 