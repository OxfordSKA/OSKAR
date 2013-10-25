/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_CONVERT_XYZ_TO_LON_LAT_H_
#define OSKAR_CONVERT_XYZ_TO_LON_LAT_H_

/**
 * @file oskar_convert_xyz_to_lon_lat.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transform Cartesian coordinates to spherical (single precision).
 *
 * @details
 * This function implicitly assumes the radius of the sphere is 1.
 *
 * @param[in]  num_points  Number of points.
 * @param[out] lon         Spherical longitude coordinate (phi), in radians.
 * @param[out] lat         Spherical latitude coordinate (theta), in radians.
 * @param[in]  x           Cartesian x coordinate of points.
 * @param[in]  y           Cartesian y coordinate of points.
 * @param[in]  z           Cartesian z coordinate of points.
 */
OSKAR_EXPORT
void oskar_convert_xyz_to_lon_lat_f(int num_points, float* lon, float* lat,
        const float* x, const float* y, const float* z);

/**
 * @brief
 * Transform Cartesian coordinates to spherical (double precision).
 *
 * @details
 * This function implicitly assumes the radius of the sphere is 1.
 *
 * @param[in]  num_points  Number of points.
 * @param[out] lon         Spherical longitude coordinate (phi), in radians.
 * @param[out] lat         Spherical latitude coordinate (theta), in radians.
 * @param[in]  x           Cartesian x coordinate of points.
 * @param[in]  y           Cartesian y coordinate of points.
 * @param[in]  z           Cartesian z coordinate of points.
 */
OSKAR_EXPORT
void oskar_convert_xyz_to_lon_lat_d(int num_points, double* lon, double* lat,
        const double* x, const double* y, const double* z);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_XYZ_TO_LON_LAT_H_ */
