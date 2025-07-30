# Unit Consistency Report

## Summary
After a comprehensive analysis of the entire codebase, I've identified and fixed several unit inconsistencies. The codebase is now consistent with **meters** as the primary unit throughout.

## Issues Found and Fixed

### 1. **Preprocessing Files**
- **Issue**: `preprocess_cpe_files.py` was converting positions to kilometers
- **Fix**: Removed conversion to kilometers, now outputs positions in meters
- **Issue**: `preprocess_cpe_files_memory_efficient.py` was also converting to kilometers
- **Fix**: Removed conversion, now keeps positions in meters

### 2. **Test Files**
- **Issue**: `test_ground_truth_quality.py` was using kilometers for Earth radius and altitude calculations
- **Fix**: Updated to use meters for calculations, but display results in kilometers for readability

### 3. **Configuration Files**
- **Status**: ✅ All configuration files are consistent
- `config.py`: Uses meters for all distance parameters
- `EARTH_RADIUS_M = 6371000.0` (meters)
- `SENSOR_MAX_RANGE_M = 1000000` (meters)
- `OSPA_CUTOFF_C = 3000.0` (meters)

### 4. **Core Simulation Files**
- **Status**: ✅ All core simulation files are consistent
- `estimation_layer.py`: Uses meters for all calculations
- `perception_layer.py`: Uses meters for all calculations
- `constellation_env.py`: Uses meters for all calculations

## Current Unit Standards

### **Positions**: Meters
- Ground truth positions: meters
- Satellite positions: meters
- Track positions: meters
- Measurement positions: meters

### **Velocities**: Meters per second (m/s)
- Ground truth velocities: m/s
- Track velocities: m/s
- Measurement range rates: m/s

### **Distances**: Meters
- Sensor ranges: meters
- OSPA cutoff: meters
- Gating thresholds: meters
- Earth radius: meters

### **Angles**: Degrees
- Azimuth: degrees
- Elevation: degrees
- Pointing angles: degrees

## Key Constants (All in Meters)

```python
EARTH_RADIUS_M = 6371000.0  # Earth radius in meters
SENSOR_MAX_RANGE_M = 1000000  # Maximum detection range in meters
OSPA_CUTOFF_C = 3000.0  # OSPA cutoff distance in meters
GRAVITATIONAL_CONSTANT = 3.986004418e14  # Earth's gravitational parameter (m³/s²)
```

## Typical Values in Meters

### **LEO Altitudes**
- Minimum: 200,000 m (200 km)
- Maximum: 2,000,000 m (2000 km)
- Typical: 400,000 - 600,000 m (400-600 km)

### **Orbital Velocities**
- Minimum: 6,000 m/s (6 km/s)
- Maximum: 9,000 m/s (9 km/s)
- Typical: 7,000 m/s (7 km/s)

### **Detection Ranges**
- Maximum: 1,000,000 m (1000 km)
- Clutter minimum: 1,000 m (1 km)

## Verification

The coordinate system fix in `estimation_layer.py` now correctly:
1. Converts measurements from relative satellite-centric coordinates to absolute ECI coordinates
2. Compares track positions (in meters) with measurement positions (in meters)
3. Uses consistent distance thresholds (1000 meters for gating)

## Conclusion

✅ **All unit inconsistencies have been resolved**
✅ **The codebase now uses meters consistently throughout**
✅ **The coordinate system fix should resolve the 0% detection rate issue**

The simulation should now work correctly with proper coordinate transformations and consistent units. 