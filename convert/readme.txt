Coordinate system specifiers
===============================================================================
* az_el
  - Left-handed spherical coordinates in the local horizontal system.
    Azimuth is measured from North to East (positive from y towards x).
    Elevation is measured from the horizon (positive towards z).

* cirs_ra_dec
  - Right-handed equatorial coordinate system used to specify celestial
    objects (East-positive):
    Celestial Intermediate Reference System Right Ascension, Declination.
    Analogous to geocentric apparent equatorial coordinates in the old system.  

* cirs_relative_directions
  - Right-handed direction cosines in CIRS, relative to a reference point
    on the sphere. Directions are l,m,n, where l is to the East,
    m is to the North, and n is towards the source.

* ecef
  - Right-handed geocentric Cartesian, Earth-Centred-Earth-Fixed coordinates.
    http://en.wikipedia.org/wiki/ECEF

* enu
  - Right-handed Cartesian vector components in the local horizontal
    system (East-North-Up). 

* enu_directions
  - Right handed Cartesian direction cosines in the local horizontal
    system (East-North-Up).

* geodetic_spherical
  - A coordinate frame for the Earth based on a standard ellipsoidal 
    reference surface (WGS84). Longitude is East-positive.

* fk5
  - Fifth fundamental catalogue. Right-handed equatorial system at J2000
    equinox.

* galactic
  - A celestial coordinate system in spherical coordinates with the Sun at its
    centre, aligned with the Galactic plane.
    http://en.wikipedia.org/wiki/Galactic_coordinate_system

* icrs_ra_dec
  - Right-handed equatorial coordinate system used to specify celestial
    objects (East-positive):
    International Celestial Reference System Right Ascension, Declination.
    Closely aligned with, but supersedes, the old J2000 system. 

* lon_lat
  - Generic right-handed spherical coordinates.
    Can apply to either the Earth or the sky.
    Longitude is the angle measured East from a prime meridian.
    Latitude is the angle measured North from the equator.

* offset_ecef
  - Cartesian coordinates with axes parallel to the ECEF frame.
    Coordinates are offset relative to some reference point on the Earth's
    surface to reduce the magnitude of the numbers.

* relative_directions
  - Right-handed generic direction cosines, relative to a reference point
    on the sphere.

* theta_phi
  - Generic right-handed spherical coordinates.
    Theta is the polar angle, phi is the azimuthal angle.
    When applied to the horizontal system, theta is the zenith angle
    (co-elevation) and phi is the co-azimuth.

Modifiers
===============================================================================
* cuda
  - Function implemented for the GPU using NVIDIA CUDA. Device kernels and
    kernel wrappers.
