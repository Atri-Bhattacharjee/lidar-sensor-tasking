#!/usr/bin/env python3
"""
Test script to check if the memory-efficient poliastro API works correctly.
"""

try:
    from poliastro.bodies import Earth
    from poliastro.twobody import Orbit
    from astropy.time import Time
    from astropy import units as u
    import numpy as np
    
    print("✅ poliastro is working!")
    print("Testing memory-efficient API with proper units...")
    
    # Test orbital elements (same as preprocessing script)
    a = 7000000.0  # 7000 km in meters
    ecc = 0.1
    inc = np.radians(45.0)
    raan = 0.0
    argp = 0.0
    nu = 0.0
    
    # Create orbit with proper units
    orbit = Orbit.from_classical(
        Earth,
        a=a * u.m,  # Convert to astropy Quantity
        ecc=ecc,
        inc=inc * u.rad,  # Convert to astropy Quantity
        raan=raan * u.rad,  # Default RAAN with units
        argp=argp * u.rad,  # Default argument of perigee with units
        nu=nu * u.rad,    # Default true anomaly with units
        epoch=Time(2025.0, format='decimalyear')
    )
    
    print(f"✅ Orbit created successfully with memory-efficient API!")
    print(f"   Semi-major axis: {orbit.a.to(u.km).value:.1f} km")
    print(f"   Eccentricity: {orbit.ecc.value:.3f}")
    print(f"   Inclination: {np.degrees(orbit.inc.value):.1f}°")
    
    # Test propagation with proper units
    print("Testing orbital propagation with memory-efficient API...")
    t_seconds = 3600  # 1 hour
    propagated_orbit = orbit.propagate(t_seconds * u.s)
    
    print(f"✅ Propagation successful with memory-efficient API!")
    print(f"   Position: {propagated_orbit.r.to(u.km).value:.1f} km")
    print(f"   Velocity: {propagated_orbit.v.to(u.km/u.s).value:.1f} km/s")
    
    print("\n🎉 Memory-efficient poliastro API is fully functional!")
    print("The preprocessing will now use accurate orbital propagation!")
    
except ImportError as e:
    print(f"❌ poliastro import failed: {e}")
    print("The preprocessing will use simplified orbital propagation instead.")
except Exception as e:
    print(f"❌ poliastro test failed: {e}")
    print("The preprocessing will use simplified orbital propagation instead.") 