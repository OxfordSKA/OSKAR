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

#ifndef OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_CUDA_H_
#define OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_CUDA_H_

/**
 * @file oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Equatorial to relative 3D direction cosines (single precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  d_ra       Input position Right Ascensions in radians.
 * @param[in]  d_dec      Input position Declinations in radians.
 * @param[in]  ra0        Right Ascension of the reference point in radians.
 * @param[in]  dec0       Declination of the reference point in radians.
 * @param[out] d_l        l-direction-cosines relative to the reference point.
 * @param[out] d_m        m-direction-cosines relative to the reference point.
 * @param[out] d_n        n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_f(
        int num_points, const float* d_ra, const float* d_dec, float ra0,
        float dec0, float* d_l, float* d_m, float* d_n);

/**
 * @brief
 * Equatorial to relative 3D direction cosines (double precision).
 *
 * @details
 * This function computes the l,m,n direction cosines of the specified points
 * relative to the reference point.
 *
 * @param[in]  num_points Number of points.
 * @param[in]  d_ra       Input position Right Ascensions in radians.
 * @param[in]  d_dec      Input position Declinations in radians.
 * @param[in]  ra0        Right Ascension of the reference point in radians.
 * @param[in]  dec0       Declination of the reference point in radians.
 * @param[out] d_l        l-direction-cosines relative to the reference point.
 * @param[out] d_m        m-direction-cosines relative to the reference point.
 * @param[out] d_n        n-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_d(
        int num_points, const double* d_ra, const double* d_dec, double ra0,
        double dec0, double* d_l, double* d_m, double* d_n);

#ifdef __CUDACC__

/**
 * @brief
 * CUDA kernel to convert spherical coordinates (RA, Dec) to direction cosines
 * centred on ra0, dec0 (single precision).
 *
 * @details
 * Computes the l,m,n direction cosines of the specified points relative to the
 * reference point.
 *
 * @param[in]  np      Number of positions.
 * @param[in]  ra      Right Ascensions, in radians.
 * @param[in]  dec     Declinations, in radians.
 * @param[in]  ra0     Centre (reference point) Right Ascension in radians.
 * @param[in]  cosDec0 Cosine of central (reference point) declination.
 * @param[in]  sinDec0 Sine of central (reference point) declination.
 * @param[out] l       Projected x-positions.
 * @param[out] m       Projected y-positions.
 * @param[out] n       Projected z-positions.
 */
__global__
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_f(
        const int np, const float* ra, const float* dec, const float ra0,
        const float cosDec0, const float sinDec0, float* l, float* m, float* n);

/**
 * @brief
 * CUDA kernel to convert spherical coordinates (RA, Dec) to direction cosines
 * centred on ra0, dec0 (double precision).
 *
 * @details
 * Computes the l,m,n direction cosines of the specified points relative to the
 * reference point.
 *
 * @param[in]  np      Number of positions.
 * @param[in]  ra      Right Ascensions, in radians.
 * @param[in]  dec     Declinations, in radians.
 * @param[in]  ra0     Centre (reference point) Right Ascension in radians.
 * @param[in]  cosDec0 Cosine of central (reference point) declination.
 * @param[in]  sinDec0 Sine of central (reference point) declination.
 * @param[out] l       Projected x-positions.
 * @param[out] m       Projected y-positions.
 * @param[out] n       Projected z-positions.
 */
__global__
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_d(
        const int np, const double* lon, const double* lat, const double lon0,
        const double cosLat0, const double sinLat0, double* l, double* m,
        double* n);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_APPARENT_RA_DEC_TO_RELATIVE_DIRECTION_COSINES_CUDA_H_ */
