/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#ifndef OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_H_
#define OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_H_

/**
 * @file oskar_convert_lon_lat_to_relative_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Spherical to relative 3D direction cosines (single precision).
 *
 * @details
 * This function computes the direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  lon_rad    Input longitudes in radians.
 * @param[in]  lat_rad    Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  cos_lat0   Cosine latitude of the reference point.
 * @param[in]  sin_lat0   Sine latitude of the reference point.
 * @param[out] l          l-direction-cosines relative to the reference point.
 * @param[out] m          m-direction-cosines relative to the reference point.
 * @param[out] n          n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_3d_f(int num_points,
        const float* lon_rad, const float* lat_rad, float lon0_rad,
        float cos_lat0, float sin_lat0, float* l, float* m, float* n);

/**
 * @brief
 * Spherical to relative 3D direction cosines (double precision).
 *
 * @details
 * This function computes the direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  lon_rad    Input longitudes in radians.
 * @param[in]  lat_rad    Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  cos_lat0   Cosine latitude of the reference point.
 * @param[in]  sin_lat0   Sine latitude of the reference point.
 * @param[out] l          l-direction-cosines relative to the reference point.
 * @param[out] m          m-direction-cosines relative to the reference point.
 * @param[out] n          n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_3d_d(int num_points,
        const double* lon_rad, const double* lat_rad, double lon0_rad,
        double cos_lat0, double sin_lat0, double* l, double* m, double* n);

/**
 * @brief
 * Spherical to relative 2D direction cosines (single precision).
 *
 * @details
 * This function computes the direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  lon_rad    Input longitudes in radians.
 * @param[in]  lat_rad    Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  cos_lat0   Cosine latitude of the reference point.
 * @param[in]  sin_lat0   Sine latitude of the reference point.
 * @param[out] l          l-direction-cosines relative to the reference point.
 * @param[out] m          m-direction-cosines relative to the reference point.
 * @param[out] n          Must be NULL.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_2d_f(int num_points,
        const float* lon_rad, const float* lat_rad, float lon0_rad,
        float cos_lat0, float sin_lat0, float* l, float* m, float* n);

/**
 * @brief
 * Spherical to relative 2D direction cosines (double precision).
 *
 * @details
 * This function computes the direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  lon_rad    Input longitudes in radians.
 * @param[in]  lat_rad    Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  cos_lat0   Cosine latitude of the reference point.
 * @param[in]  sin_lat0   Sine latitude of the reference point.
 * @param[out] l          l-direction-cosines relative to the reference point.
 * @param[out] m          m-direction-cosines relative to the reference point.
 * @param[out] n          Must be NULL.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_2d_d(int num_points,
        const double* lon_rad, const double* lat_rad, double lon0_rad,
        double cos_lat0, double sin_lat0, double* l, double* m, double* n);

/**
 * @brief
 * Spherical to relative 2D or 3D direction cosines.
 *
 * @details
 * This function computes the direction cosines of the specified points
 * relative to the reference point.
 *
 * If only a 2D transform is required, \p n can be NULL on input.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  lon_rad    Input longitudes in radians.
 * @param[in]  lat_rad    Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  lat0_rad   Latitude of the reference point in radians.
 * @param[out] l          l-direction-cosines relative to the reference point.
 * @param[out] m          m-direction-cosines relative to the reference point.
 * @param[out] n          n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions(int num_points,
        const oskar_Mem* lon_rad, const oskar_Mem* lat_rad, double lon0_rad,
        double lat0_rad, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_H_ */
