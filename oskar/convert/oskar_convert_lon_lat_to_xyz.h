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

#ifndef OSKAR_CONVERT_LON_LAT_TO_XYZ_H_
#define OSKAR_CONVERT_LON_LAT_TO_XYZ_H_

/**
 * @file oskar_convert_lon_lat_to_xyz.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transform spherical coordinates to Cartesian (single precision).
 *
 * @details
 * Transform spherical coordinates to Cartesian on the unit sphere.
 *
 * @param[in]  num_points  Number of points.
 * @param[in]  lon_rad     Input longitude coordinates, in radians.
 * @param[in]  lat_rad     Input latitude coordinates, in radians.
 * @param[out] x           Cartesian x coordinate of points.
 * @param[out] y           Cartesian y coordinate of points.
 * @param[out] z           Cartesian z coordinate of points.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_xyz_f(int num_points, const float* lon_rad,
        const float* lat_rad, float* x, float* y, float* z);

/**
 * @brief
 * Transform spherical coordinates to Cartesian (double precision).
 *
 * @details
 * Transform spherical coordinates to Cartesian on the unit sphere.
 *
 * @param[in]  num_points  Number of points.
 * @param[in]  lon_rad     Input longitude coordinates, in radians.
 * @param[in]  lat_rad     Input latitude coordinates, in radians.
 * @param[out] x           Cartesian x coordinate of points.
 * @param[out] y           Cartesian y coordinate of points.
 * @param[out] z           Cartesian z coordinate of points.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_xyz_d(int num_points, const double* lon_rad,
        const double* lat_rad, double* x, double* y, double* z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_LON_LAT_TO_XYZ_H_ */
