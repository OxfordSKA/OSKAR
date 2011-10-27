/*
 * Copyright (c) 2011, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_HORIZON_PLANE_TO_GEOCENTRIC_CARTESIAN_H_
#define OSKAR_HORIZON_PLANE_TO_GEOCENTRIC_CARTESIAN_H_

/**
 * @file oskar_horizon_plane_to_geocentric_cartesian.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from horizon plane to geocentric cartesian system.
 * (double precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to the geocentric cartesian (Earth-Centred-
 * Earth-Fixed, or ECEF) system.
 *
 * The reference latitude is, strictly speaking, geodetic.
 *
 * The input coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing to the local East, the y-axis to the local
 * North, and the z-axis to the zenith.
 *
 * The output coordinates are with respect to the origin at the centre of the
 * Earth, and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * @param[in]  num_antennas  Number of antennas / stations.
 * @param[in]  x_horizon     Vector of horizontal station x-positions, in metres.
 * @param[in]  y_horizon     Vector of horizontal station y-positions, in metres.
 * @param[in]  z_horizon     Vector of horizontal station z-positions, in metres.
 * @param[in]  longitude     Telescope longitude, in radians.
 * @param[in]  latitude      Telescope latitude, in radians.
 * @param[in]  altitude      Telescope altitude above ellipsoid, in metres.
 * @param[out] x             Vector of ECEF x-positions, in metres.
 * @param[out] y             Vector of ECEF y-positions, in metres.
 * @param[out] z             Vector of ECEF z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_horizon_plane_to_geocentric_cartesian_d(int num_antennas,
        const double* x_horizon, const double* y_horizon,
        const double* z_horizon, double longitude, double latitude,
        double altitude, double* x, double* y, double* z);

/**
 * @brief
 * Convert coordinates from horizon plane to geocentric cartesian system
 * (single precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to the geocentric cartesian (Earth-Centred-
 * Earth-Fixed, or ECEF) system.
 *
 * The reference latitude is, strictly speaking, geodetic.
 *
 * The input coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing to the local East, the y-axis to the local
 * North, and the z-axis to the zenith.
 *
 * The output coordinates are with respect to the origin at the centre of the
 * Earth, and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * @param[in]  num_antennas  Number of antennas / stations.
 * @param[in]  x_horizon     Vector of horizontal station x-positions, in metres.
 * @param[in]  y_horizon     Vector of horizontal station y-positions, in metres.
 * @param[in]  z_horizon     Vector of horizontal station z-positions, in metres.
 * @param[in]  longitude     Telescope longitude, in radians.
 * @param[in]  latitude      Telescope latitude, in radians.
 * @param[in]  altitude      Telescope altitude above ellipsoid, in metres.
 * @param[out] x             Vector of ECEF x-positions, in metres.
 * @param[out] y             Vector of ECEF y-positions, in metres.
 * @param[out] z             Vector of ECEF z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_horizon_plane_to_geocentric_cartesian_f(int num_antennas,
        const float* x_horizon, const float* y_horizon,
        const float* z_horizon, float longitude, float latitude,
        float altitude, float* x, float* y, float* z);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_HORIZON_PLANE_TO_GEOCENTRIC_CARTESIAN_H_
