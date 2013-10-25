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

#ifndef OSKAR_CONVERT_LON_LAT_TO_TANGENT_PLANE_DIRECTION_H_
#define OSKAR_CONVERT_LON_LAT_TO_TANGENT_PLANE_DIRECTION_H_

/**
 * @file oskar_convert_lon_lat_to_tangent_plane_direction.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Project spherical coordinates (single precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * For normal fields of view at normal latitudes (i.e. not at the poles),
 * the minimum and maximum values of x and y correspond to
 * the minimum and maximum values of the longitude and latitude, respectively.
 *
 * @param[in]  num_positions Number of positions.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  lon           Array of longitude values, in radians.
 * @param[in]  lat           Array of latitude values, in radians.
 * @param[out] x             Array of x-positions in cosine space.
 * @param[out] y             Array of y-positions in cosine space.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_tangent_plane_direction_f(int num_positions,
        float lon0, float lat0, const float* lon, const float* lat, float* x,
        float* y);

/**
 * @brief
 * Project spherical coordinates (double precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * For normal fields of view at normal latitudes (i.e. not at the poles),
 * the minimum and maximum values of x and y correspond to
 * the minimum and maximum values of the longitude and latitude, respectively.
 *
 * @param[in]  num_positions Number of positions.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  lon           Array of longitude values, in radians.
 * @param[in]  lat           Array of latitude values, in radians.
 * @param[out] x             Array of x-positions in cosine space.
 * @param[out] y             Array of y-positions in cosine space.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_tangent_plane_direction_d(int num_positions,
        double lon0, double lat0, const double* lon, const double* lat,
        double* x, double* y);

/**
 * @brief
 * Project spherical coordinates using OpenMP (single precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * For normal fields of view at normal latitudes (i.e. not at the poles),
 * the minimum and maximum values of x and y correspond to
 * the minimum and maximum values of the longitude and latitude, respectively.
 *
 * @param[in]  num_positions Number of positions.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  lon           Array of longitude values, in radians.
 * @param[in]  lat           Array of latitude values, in radians.
 * @param[out] x             Array of x-positions in cosine space.
 * @param[out] y             Array of y-positions in cosine space.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_tangent_plane_direction_omp_f(int num_positions,
        float lon0, float lat0, const float* lon, const float* lat, float* x,
        float* y);

/**
 * @brief
 * Project spherical coordinates using OpenMP (double precision).
 *
 * @details
 * Projects spherical coordinates at the specified tangent point using the
 * orthographic tangent-plane projection.
 *
 * For normal fields of view at normal latitudes (i.e. not at the poles),
 * the minimum and maximum values of x and y correspond to
 * the minimum and maximum values of the longitude and latitude, respectively.
 *
 * @param[in]  num_positions Number of positions.
 * @param[in]  lon0          Longitude of the field centre, in radians.
 * @param[in]  lat0          Latitude of the field centre, in radians.
 * @param[in]  lon           Array of longitude values, in radians.
 * @param[in]  lat           Array of latitude values, in radians.
 * @param[out] x             Array of x-positions in cosine space.
 * @param[out] y             Array of y-positions in cosine space.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_tangent_plane_direction_omp_d(int num_positions,
        double lon0, double lat0, const double* lon, const double* lat,
        double* x, double* y);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_LON_LAT_TO_TANGENT_PLANE_DIRECTION_H_ */
