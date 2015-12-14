/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_CONVERT_ENU_TO_OFFSET_ECEF_H_
#define OSKAR_CONVERT_ENU_TO_OFFSET_ECEF_H_

/**
 * @file oskar_convert_enu_to_offset_ecef.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from horizon plane (ENU) to offset geocentric Cartesian
 * (double precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to an offset geocentric Cartesian system.
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
 * @param[in]  num_points    Number of points.
 * @param[in]  horizon_x     Horizontal x-positions (east), in metres.
 * @param[in]  horizon_y     Horizontal y-positions (north), in metres.
 * @param[in]  horizon_z     Horizontal z-positions (up), in metres.
 * @param[in]  lon_rad       Longitude of tangent point, in radians.
 * @param[in]  lat_rad       Latitude of tangent point, in radians.
 * @param[out] offset_ecef_x Vector of output x-positions, in metres.
 * @param[out] offset_ecef_y Vector of output y-positions, in metres.
 * @param[out] offset_ecef_z Vector of output z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_convert_enu_to_offset_ecef_d(int num_points,
        const double* horizon_x, const double* horizon_y,
        const double* horizon_z, double lon_rad, double lat_rad,
        double* offset_ecef_x, double* offset_ecef_y, double* offset_ecef_z);

/**
 * @brief
 * Convert coordinates from horizon plane (ENU) to offset geocentric Cartesian
 * (single precision).
 *
 * @details
 * This function converts station positions from the local horizon plane
 * (East-North-Up, or ENU) to an offset geocentric Cartesian system.
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
 * @param[in]  num_points    Number of points.
 * @param[in]  horizon_x     Horizontal x-positions (east), in metres.
 * @param[in]  horizon_y     Horizontal y-positions (north), in metres.
 * @param[in]  horizon_z     Horizontal z-positions (up), in metres.
 * @param[in]  lon_rad       Longitude of tangent point, in radians.
 * @param[in]  lat_rad       Latitude of tangent point, in radians.
 * @param[out] offset_ecef_x Vector of output x-positions, in metres.
 * @param[out] offset_ecef_y Vector of output y-positions, in metres.
 * @param[out] offset_ecef_z Vector of output z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_convert_enu_to_offset_ecef_f(int num_points,
        const float* horizon_x, const float* horizon_y,
        const float* horizon_z, float lon_rad, float lat_rad,
        float* offset_ecef_x, float* offset_ecef_y, float* offsec_ecef_z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ENU_TO_OFFSET_ECEF_H_ */
