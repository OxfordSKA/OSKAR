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

#ifndef OSKAR_OFFSET_GEOCENTRIC_CARTESIAN_TO_GEOCENTRIC_CARTESIAN_H_
#define OSKAR_OFFSET_GEOCENTRIC_CARTESIAN_TO_GEOCENTRIC_CARTESIAN_H_

/**
 * @file oskar_offset_geocentric_cartesian_to_geocentric_cartesian.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert coordinates from the offset geocentric cartesian to the geocentric
 * cartesian system.
 *
 * @details
 * This function converts station positions from the offset geocentric
 * cartesian to the geocentric cartesian (Earth-Centred-
 * Earth-Fixed, or ECEF) system.
 *
 * The reference latitude is, strictly speaking, geodetic.
 *
 * The input coordinates are with respect to the origin at the tangent point,
 * and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * The output coordinates are with respect to the origin at the centre of the
 * Earth, and have the x-axis pointing towards the meridian of zero longitude,
 * the y-axis to +90 degrees East, and the z-axis to the North Celestial Pole.
 *
 * A single precision version of this function is not provided, because it
 * would be unable to represent points accurately on the Earth's surface
 * (more than 7 decimal digits are required for sub-metre precision).
 *
 * @param[in]  n             Number of points.
 * @param[in]  x_offset      Vector of offset x-positions, in metres.
 * @param[in]  y_offset      Vector of offset y-positions, in metres.
 * @param[in]  z_offset      Vector of offset z-positions, in metres.
 * @param[in]  longitude     Longitude of tangent point, in radians.
 * @param[in]  latitude      Latitude of tangent point, in radians.
 * @param[in]  altitude      Altitude above ellipsoid, in metres.
 * @param[out] x             Vector of ECEF x-positions, in metres.
 * @param[out] y             Vector of ECEF y-positions, in metres.
 * @param[out] z             Vector of ECEF z-positions, in metres.
 */
OSKAR_EXPORT
void oskar_offset_geocentric_cartesian_to_geocentric_cartesian(int n,
        const double* x_offset, const double* y_offset,
        const double* z_offset, double longitude, double latitude,
        double altitude, double* x, double* y, double* z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_OFFSET_GEOCENTRIC_CARTESIAN_TO_GEOCENTRIC_CARTESIAN_H_ */
