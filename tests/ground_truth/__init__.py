"""Ground truth tests for thistle.orbit_data using verified reference data.

This package contains comprehensive validation tests organized by data type:
- test_eci.py: ECI position and velocity
- test_lla.py: Latitude, longitude, altitude
- test_keplerian.py: Keplerian orbital elements
- test_magnetic_field.py: IGRF magnetic field model
- test_beta_angle.py: Beta angle (Sun angle to orbit plane)
- test_sunlight.py: Sunlight/eclipse detection
- test_orbital_mechanics.py: Physics-based validation tests

Ground truth data source: STK (Systems Tool Kit) propagation using
high-fidelity HPOP propagator for ISS on April 1, 2020.
"""
