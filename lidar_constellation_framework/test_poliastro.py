#!/usr/bin/env python3
"""
Test script to check if poliastro is working correctly.
"""

try:
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    print("✅ poliastro is working!")
    
    # Test basic functionality
    print("Testing basic orbital creation...")
    
    # Create a simple orbit
    from astropy.time import Time
    import numpy as np
    
    # Test orbital elements
    a = 7000000.0  # 7000 km in meters
    ecc = 0.1
    inc = np.radians(45.0)
    raan = 0.0
    argp = 0.0
    nu = 0.0
    
    # Create orbit
    orbit = Orbit.from_classical(
        Earth,
        a=a,
        ecc=ecc,
        inc=inc,
        raan=raan,
        argp=argp,
        nu=nu,
        epoch=Time(2025.0, format='decimalyear')
    )
    
    print(f"✅ Orbit created successfully!")
    print(f"   Semi-major axis: {orbit.a.value/1000:.1f} km")
    print(f"   Eccentricity: {orbit.ecc.value:.3f}")
    print(f"   Inclination: {np.degrees(orbit.inc.value):.1f}°")
    
    # Test propagation
    print("Testing orbital propagation...")
    propagated_orbit = orbit.propagate(3600)  # 1 hour
    
    print(f"✅ Propagation successful!")
    print(f"   Position: {propagated_orbit.r.value/1000:.1f} km")
    print(f"   Velocity: {propagated_orbit.v.value/1000:.1f} km/s")
    
    print("\n🎉 poliastro is fully functional and ready for preprocessing!")
    
except ImportError as e:
    print(f"❌ poliastro import failed: {e}")
    print("The preprocessing will use simplified orbital propagation instead.")
except Exception as e:
    print(f"❌ poliastro test failed: {e}")
    print("The preprocessing will use simplified orbital propagation instead.") 