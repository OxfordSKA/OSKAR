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

#ifndef OSKAR_HORIZON_PLANE_TO_OFFSET_GEOCENTRIC_CARTESIAN_H_
#define OSKAR_HORIZON_PLANE_TO_OFFSET_GEOCENTRIC_CARTESIAN_H_

/**
 * @file oskar_horizon_plane_to_offset_geocentric_cartesian.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from horizon plane to offset geocentric cartesian system
 * (double precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to an offset geocentric cartesian system.
 * The "offset" is because both systems share a common origin (not at the
 * centre of the Earth), thus preserving the precision of the input data.
 *
 * The reference latitude is, strictly speaking, geodetic.
 *
 * The input coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing to the local East, the y-axis to the local
 * North, and the z-axis to the zenith.
 *
 * The output coordinates are with respect to the same origin, and have the
 * x-axis pointing towards the meridian of zero longitude, the y-axis to +90
 * degrees East, and the z-axis to the North Celestial Pole.
 *
 * @param[in]  n             Number of points.
 * @param[in]  x_horizon     Vector of horizontal x-positions, in metres.
 * @param[in]  y_horizon     Vector of horizontal y-positions, in metres.
 * @param[in]  z_horizon     Vector of horizontal z-positions, in metres.
 * @param[in]  longitude     Longitude of tangent point, in radians.
 * @param[in]  latitude      Latitude of tangent point, in radians.
 * @param[in]  altitude      Altitude above ellipsoid, in metres.
 * @param[out] x             Vector of output x-positions, in metres.
 * @param[out] y             Vector of output y-positions, in metres.
 * @param[out] z             Vector of output z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_horizon_plane_to_offset_geocentric_cartesian_d(int n,
        const double* x_horizon, const double* y_horizon,
        const double* z_horizon, double longitude, double latitude,
        double altitude, double* x, double* y, double* z);

/**
 * @brief
 * Convert coordinates from horizon plane to offset geocentric cartesian system
 * (single precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to an offset geocentric cartesian system.
 * The "offset" is because both systems share a common origin (not at the
 * centre of the Earth), thus preserving the precision of the input data.
 *
 * The reference latitude is, strictly speaking, geodetic.
 *
 * The input coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing to the local East, the y-axis to the local
 * North, and the z-axis to the zenith.
 *
 * The output coordinates are with respect to the same origin, and have the
 * x-axis pointing towards the meridian of zero longitude, the y-axis to +90
 * degrees East, and the z-axis to the North Celestial Pole.
 *
 * @param[in]  n             Number of points.
 * @param[in]  x_horizon     Vector of horizontal x-positions, in metres.
 * @param[in]  y_horizon     Vector of horizontal y-positions, in metres.
 * @param[in]  z_horizon     Vector of horizontal z-positions, in metres.
 * @param[in]  longitude     Longitude of tangent point, in radians.
 * @param[in]  latitude      Latitude of tangent point, in radians.
 * @param[in]  altitude      Altitude above ellipsoid, in metres.
 * @param[out] x             Vector of output x-positions, in metres.
 * @param[out] y             Vector of output y-positions, in metres.
 * @param[out] z             Vector of output z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_horizon_plane_to_offset_geocentric_cartesian_f(int n,
        const float* x_horizon, const float* y_horizon,
        const float* z_horizon, float longitude, float latitude,
        float altitude, float* x, float* y, float* z);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_HORIZON_PLANE_TO_OFFSET_GEOCENTRIC_CARTESIAN_H_
