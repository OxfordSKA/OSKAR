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

#ifndef OSKAR_CONVERT_ECEF_TO_STATION_UVW_CUDA_H_
#define OSKAR_CONVERT_ECEF_TO_STATION_UVW_CUDA_H_

/**
 * @file oskar_convert_ecef_to_station_uvw_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates using CUDA
 * (single precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this CUDA
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  num_stations The size of the station coordinate arrays.
 * @param[in]  d_x          Input station x coordinates (ECEF or related frame).
 * @param[in]  d_y          Input station y coordinates (ECEF or related frame).
 * @param[in]  d_z          Input station z coordinates (ECEF or related frame).
 * @param[in]  ha0_rad      The Hour Angle of the phase centre, in radians.
 * @param[in]  dec0_rad     The Declination of the phase centre, in radians.
 * @param[out] d_u          Output station u coordinates.
 * @param[out] d_v          Output station v coordinates.
 * @param[out] d_w          Output station w coordinates.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_station_uvw_cuda_f(int num_stations,
        const float* d_x, const float* d_y, const float* d_z,
        float ha0_rad, float dec0_rad, float* d_u, float* d_v, float* d_w);

/**
 * @brief
 * Transforms station (x,y,z) coordinates to (u,v,w) coordinates using CUDA
 * (double precision).
 *
 * @details
 * Given the hour angle and declination of the phase tracking centre, this CUDA
 * function transforms the station (x,y,z) coordinates to (u,v,w) coordinates.
 *
 * @param[in]  num_stations The size of the station coordinate arrays.
 * @param[in]  d_x          Input station x coordinates (ECEF or related frame).
 * @param[in]  d_y          Input station y coordinates (ECEF or related frame).
 * @param[in]  d_z          Input station z coordinates (ECEF or related frame).
 * @param[in]  ha0_rad      The Hour Angle of the phase centre, in radians.
 * @param[in]  dec0_rad     The Declination of the phase centre, in radians.
 * @param[out] d_u          Output station u coordinates.
 * @param[out] d_v          Output station v coordinates.
 * @param[out] d_w          Output station w coordinates.
 */
OSKAR_EXPORT
void oskar_convert_ecef_to_station_uvw_cuda_d(int num_stations,
        const double* d_x, const double* d_y, const double* d_z,
        double ha0_rad, double dec0_rad, double* d_u, double* d_v, double* d_w);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_convert_ecef_to_station_uvw_cudak_f(const int num_stations,
        const float* restrict x, const float* restrict y,
        const float* restrict z, const float sin_ha0,
        const float cos_ha0, const float sin_dec0, const float cos_dec0,
        float* restrict u, float* restrict v, float* restrict w);

__global__
void oskar_convert_ecef_to_station_uvw_cudak_d(const int num_stations,
        const double* restrict x, const double* restrict y,
        const double* restrict z, const double sin_ha0,
        const double cos_ha0, const double sin_dec0, const double cos_dec0,
        double* restrict u, double* restrict v, double* restrict w);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ECEF_TO_STATION_UVW_CUDA_H_ */
