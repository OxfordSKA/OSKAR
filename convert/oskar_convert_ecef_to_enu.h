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

#ifndef OSKAR_CONVERT_ECEF_TO_ENU_H_
#define OSKAR_CONVERT_ECEF_TO_ENU_H_

/**
 * @file oskar_convert_ecef_to_enu.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from the Earth-Centred-Earth-Fixed (ECEF) to a local
 * horizon system.
 *
 * @details
 * This function converts station positions from the geocentric Cartesian
 * (Earth-Centred-Earth-Fixed, or ECEF) system to the local horizon plane
 * (East-North-Up, or ENU).
 *
 * The input coordinates are with respect to the origin at the centre of the
 * Earth, and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * The output coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing to the local East, the y-axis to the local
 * North, and the z-axis to the zenith.
 *
 * A single precision version of this function is not provided, because it
 * would be unable to represent points accurately on the Earth's surface
 * (more than 7 decimal digits are required for sub-metre precision).
 *
 * @param[in]  num_points Number of points.
 * @param[in]  ecef_x     ECEF x-positions, in metres.
 * @param[in]  ecef_y     ECEF y-positions, in metres.
 * @param[in]  ecef_z     ECEF z-positions, in metres.
 * @param[in]  lon_rad    Longitude of reference point, in radians.
 * @param[in]  lat_rad    Latitude of reference point, in radians.
 * @param[in]  alt_metres Altitude of reference point from ellipsoid, in metres.
 * @param[out] x          Horizontal x-positions (east), in metres.
 * @param[out] y          Horizontal y-positions (north), in metres.
 * @param[out] z          Horizontal z-positions (up), in metres.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_enu(int num_points, const double* ecef_x,
        const double* ecef_y, const double* ecef_z, double lon_rad,
        double lat_rad, double alt_metres, double* x, double* y, double* z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_HORIZON_H_ */
