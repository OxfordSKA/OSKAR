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

#ifndef OSKAR_CONVERT_APPARENT_HA_DEC_TO_ENU_DIRECTION_COSINES_CUDA_H_
#define OSKAR_CONVERT_APPARENT_HA_DEC_TO_ENU_DIRECTION_COSINES_CUDA_H_

/**
 * @file oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __CUDACC__

/**
 * @brief
 * CUDA kernel to convert equatorial coordinates to horizontal direction cosines
 * (single precision).
 *
 * @details
 * This CUDA kernel transforms coordinates specified in a local equatorial
 * system (HA, Dec) to direction cosines l,m,n in the horizontal system.
 *
 * The direction cosines in the horizon frame are unit vectors with axes:
 * <li> x - pointing East, </li>
 * <li> y - pointing North, </li>
 * <li> z - pointing to the zenith. </li>
 *
 * If the source n-value is less than 0, then the source is below the horizon.
 *
 * @param[in]  ns     The number of positions.
 * @param[in]  ha     Hour Angles in radians.
 * @param[in]  dec    Declinations in radians.
 * @param[in]  cosLat The cosine of the geodetic latitude of the origin of the
 *                    horizon frame.
 * @param[in]  sinLat The sine of the geodetic latitude of the origin of the
 *                    horizon frame.
 * @param[out] x      x direction cosines.
- * @param[out] y      y direction cosines
 * @param[out] z      z direction cosines
 */
__global__
void oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_f(int n,
        const float* ha, const float* dec, float cosLat, float sinLat,
        float* x, float* y, float* z);

/**
 * @brief
 * CUDA kernel to convert equatorial coordinates to horizontal direction cosines
 * (double precision).
 *
 * @details
 * This CUDA kernel transforms coordinates specified in a local equatorial
 * system (HA, Dec) to direction cosines l,m,n in the horizontal system.
 *
 * The direction cosines in the horizon frame are unit vectors with axes:
 * <li> x - pointing East, </li>
 * <li> y - pointing North, </li>
 * <li> z - pointing to the zenith. </li>
 *
 * If the source n-value is less than 0, then the source is below the horizon.
 *
 * @param[in]  ns     The number of positions.
 * @param[in]  ha     Hour Angles in radians.
 * @param[in]  dec    Declinations in radians.
 * @param[in]  cosLat The cosine of the geodetic latitude of the origin of the
 *                    horizon frame.
 * @param[in]  sinLat The sine of the geodetic latitude of the origin of the
 *                    horizon frame.
 * @param[out] x      x direction cosines.
 * @param[out] y      y direction cosines
 * @param[out] z      z direction cosines
 */
__global__
void oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_d(int n,
        const double* ha, const double* dec, double cosLat, double sinLat,
        double* x, double* y, double* z);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_APPARENT_HA_DEC_TO_ENU_DIRECTION_COSINES_CUDA_H_ */
