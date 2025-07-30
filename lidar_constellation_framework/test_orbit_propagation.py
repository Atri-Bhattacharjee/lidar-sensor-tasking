#!/usr/bin/env python3
"""
Test script for orbit propagation

This script tests the time-dependent orbit propagation to ensure satellites
move correctly along their orbits during the simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the simulation directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'simulation'))

from simulation.utils.constellation_loader import (
    load_constellation_parameters, 
    get_constellation_positions,
    get_constellation_summary
)
import config


def test_orbit_propagation():
    """Test the orbit propagation functionality."""
    print("=" * 60)
    print("Testing Orbit Propagation")
    print("=" * 60)
    
    # Test 1: Load constellation parameters
    print("\n1. Loading constellation parameters...")
    constellation_params = load_constellation_parameters()
    print(f"✓ Loaded {len(constellation_params)} satellites")
    
    # Test 2: Get constellation summary
    print("\n2. Getting constellation summary...")
    summary = get_constellation_summary()
    print(f"✓ Total satellites: {summary['total_satellites']}")
    print(f"✓ Number of planes: {summary['num_planes']}")
    print(f"✓ Satellites per plane: {summary['satellites_per_plane']}")
    print(f"✓ Orbital period: {summary['orbital_period_minutes']:.1f} minutes")
    print(f"✓ Altitude range: {summary['altitude_range_km'][0]:.1f} - {summary['altitude_range_km'][1]:.1f} km")
    
    # Test 3: Check initial positions
    print("\n3. Checking initial positions...")
    initial_positions = get_constellation_positions(0.0)
    print(f"✓ Initial positions shape: {initial_positions.shape}")
    print(f"✓ First satellite position: {initial_positions[0]}")
    
    # Test 4: Check positions after one orbital period
    print("\n4. Checking positions after one orbital period...")
    orbital_period_seconds = summary['orbital_period_minutes'] * 60
    final_positions = get_constellation_positions(orbital_period_seconds)
    print(f"✓ Final positions shape: {final_positions.shape}")
    print(f"✓ First satellite position after one period: {final_positions[0]}")
    
    # Test 5: Check that satellites have moved
    print("\n5. Checking satellite movement...")
    position_differences = np.linalg.norm(final_positions - initial_positions, axis=1)
    print(f"✓ Average position change: {np.mean(position_differences):.1f} meters")
    print(f"✓ Maximum position change: {np.max(position_differences):.1f} meters")
    print(f"✓ Minimum position change: {np.min(position_differences):.1f} meters")
    
    # Verify that satellites have moved significantly (should be > 1000 km for one period)
    if np.mean(position_differences) > 1000000:  # 1000 km
        print("✓ Satellites are moving correctly along their orbits")
    else:
        print("✗ Satellites are not moving enough - check propagation")
        return False
    
    # Test 6: Check intermediate positions
    print("\n6. Checking intermediate positions...")
    quarter_period = orbital_period_seconds / 4
    quarter_positions = get_constellation_positions(quarter_period)
    
    # Check that positions are different at different times
    quarter_diff = np.linalg.norm(quarter_positions - initial_positions, axis=1)
    print(f"✓ Position change at 1/4 period: {np.mean(quarter_diff):.1f} meters")
    
    # Test 7: Verify orbital mechanics
    print("\n7. Verifying orbital mechanics...")
    
    # Check that the first satellite returns to approximately the same position after one period
    first_sat_diff = np.linalg.norm(final_positions[0] - initial_positions[0])
    print(f"✓ First satellite position difference after one period: {first_sat_diff:.1f} meters")
    
    # For near-circular orbits, the position should be very close after one period
    if first_sat_diff < 10000:  # 10 km tolerance
        print("✓ Orbital period calculation is correct")
    else:
        print("✗ Orbital period calculation may be incorrect")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All orbit propagation tests passed!")
    print("=" * 60)
    return True


def visualize_orbit_propagation():
    """Create a simple visualization of orbit propagation."""
    print("\nCreating orbit propagation visualization...")
    
    # Get constellation parameters
    constellation_params = load_constellation_parameters()
    summary = get_constellation_summary()
    orbital_period_seconds = summary['orbital_period_minutes'] * 60
    
    # Sample a few satellites from different planes
    sample_satellites = ['P1-S1', 'P2-S1', 'P3-S1', 'P4-S1', 'P5-S1']
    
    # Create time array for one orbital period
    time_points = 100
    times = np.linspace(0, orbital_period_seconds, time_points)
    
    # Track positions for each sample satellite
    satellite_trajectories = {}
    
    for sat_id in sample_satellites:
        if sat_id in constellation_params:
            positions = []
            for t in times:
                pos = get_constellation_positions(t)
                # Find the index of this satellite
                sat_index = list(constellation_params.keys()).index(sat_id)
                positions.append(pos[sat_index])
            satellite_trajectories[sat_id] = np.array(positions)
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Earth (simplified as a sphere)
    earth_radius = config.EARTH_RADIUS_M / 1000  # Convert to km for plotting
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='lightblue', alpha=0.3)
    
    # Plot satellite trajectories
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (sat_id, trajectory) in enumerate(satellite_trajectories.items()):
        # Convert to km for plotting
        trajectory_km = trajectory / 1000
        ax.plot(trajectory_km[:, 0], trajectory_km[:, 1], trajectory_km[:, 2], 
                color=colors[i], label=sat_id, linewidth=2)
        
        # Mark start and end points
        ax.scatter(trajectory_km[0, 0], trajectory_km[0, 1], trajectory_km[0, 2], 
                  color=colors[i], s=50, marker='o')
        ax.scatter(trajectory_km[-1, 0], trajectory_km[-1, 1], trajectory_km[-1, 2], 
                  color=colors[i], s=50, marker='s')
    
    # Set plot properties
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.set_title('Satellite Orbit Propagation (One Orbital Period)')
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([trajectory.max() for trajectory in satellite_trajectories.values()]).max() / 1000
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.savefig('orbit_propagation_visualization.png', dpi=300, bbox_inches='tight')
    print("✓ Visualization saved as 'orbit_propagation_visualization.png'")


if __name__ == "__main__":
    try:
        # Run tests
        success = test_orbit_propagation()
        
        if success:
            # Create visualization
            visualize_orbit_propagation()
        else:
            print("\n✗ Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 