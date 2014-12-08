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

#ifndef OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_CUDA_H_
#define OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_CUDA_H_

/**
 * @file oskar_convert_lon_lat_to_relative_directions_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Spherical to relative 3D direction cosines (single precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  d_lon_rad  Input longitudes in radians.
 * @param[in]  d_lat_rad  Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  lat0_rad   Latitude of the reference point in radians.
 * @param[out] d_l        l-direction-cosines relative to the reference point.
 * @param[out] d_m        m-direction-cosines relative to the reference point.
 * @param[out] d_n        n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_cuda_f(int num_points,
        const float* d_lon_rad, const float* d_lat_rad, float lon0_rad,
        float lat0_rad, float* d_l, float* d_m, float* d_n);

/**
 * @brief
 * Spherical to relative 3D direction cosines (double precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  d_lon_rad  Input longitudes in radians.
 * @param[in]  d_lat_rad  Input latitudes in radians.
 * @param[in]  lon0_rad   Longitude of the reference point in radians.
 * @param[in]  lat0_rad   Latitude of the reference point in radians.
 * @param[out] d_l        l-direction-cosines relative to the reference point.
 * @param[out] d_m        m-direction-cosines relative to the reference point.
 * @param[out] d_n        n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_lon_lat_to_relative_directions_cuda_d(int num_points,
        const double* d_lon_rad, const double* d_lat_rad, double lon0_rad,
        double lat0_rad, double* d_l, double* d_m, double* d_n);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_convert_lon_lat_to_relative_directions_cudak_f(const int num_points,
        const float* restrict lon_rad, const float* restrict lat_rad,
        const float lon0_rad, const float cos_lat0, const float sin_lat0,
        float* restrict l, float* restrict m, float* restrict n);

__global__
void oskar_convert_lon_lat_to_relative_directions_cudak_d(const int num_points,
        const double* restrict lon_rad, const double* restrict lat_rad,
        const double lon0_rad, const double cos_lat0, const double sin_lat0,
        double* restrict l, double* restrict m, double* restrict n);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_LON_LAT_TO_RELATIVE_DIRECTIONS_CUDA_H_ */
