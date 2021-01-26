/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_ECEF_TO_GEODETIC_SPHERICAL_H_
#define OSKAR_CONVERT_ECEF_TO_GEODETIC_SPHERICAL_H_

/**
 * @file oskar_convert_ecef_to_geodetic_spherical.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from a geocentric cartesian (ECEF) to geodetic spherical
 * system.
 *
 * @details
 * This function converts positions from the geocentric cartesian
 * (Earth-Centred-Earth-Fixed, or ECEF) to the geodetic spherical (longitude,
 * latitude, altitude above ellipsoid) system.
 *
 * The function uses WGS84 ellipsoid parameters:
 * Equatorial radius: 6378137 metres.
 * Polar radius: 6356752.314 metres.
 *
 * The input coordinates are with respect to the origin at the centre of the
 * Earth, and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * The algorithm is from H. Vermeille (2002): Journal of Geodesy, 76, 451.
 * DOI 10.1007/s00190-002-0273-6
 *
 * @param[in]  num_points Number of points.
 * @param[in]  x          Vector of ECEF x-positions, in metres.
 * @param[in]  y          Vector of ECEF y-positions, in metres.
 * @param[in]  z          Vector of ECEF z-positions, in metres.
 * @param[out] lon_rad    Vector of longitudes, in radians.
 * @param[out] lat_rad    Vector of latitudes, in radians.
 * @param[out] alt_m      Vector of altitudes above ellipsoid, in metres.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_geodetic_spherical(int num_points,
        const double* x, const double* y, const double* z,
        double* lon_rad, double* lat_rad, double* alt_m);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
