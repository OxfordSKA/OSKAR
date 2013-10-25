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

#ifndef OSKAR_CONVERT_APPARENT_RA_DEC_TO_TANGENT_PLANE_DIRECTION_CUDA_H_
#define OSKAR_CONVERT_APPARENT_RA_DEC_TO_TANGENT_PLANE_DIRECTION_CUDA_H_

/**
 * @file oskar_convert_apparent_ra_dec_to_tangent_plane_direction_cuda.h
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
 * This function computes the x,y,z direction cosines of the specified points
 * relative to the reference point.
 *
 * Note that the z-positions are given by sqrt(1 - x*x - y*y) - 1.
 *
 * @param[in]  n     The number of points.
 * @param[in]  ra    Input position Right Ascensions in radians.
 * @param[in]  dec   Input position Declinations in radians.
 * @param[in]  ra0   Right Ascension of the reference point in radians.
 * @param[in]  dec0  Declination of the reference point in radians.
 * @param[out] x     x-direction-cosines relative to the reference point.
 * @param[out] y     y-direction-cosines relative to the reference point.
 * @param[out] z     z-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_tangent_plane_direction_cuda_f(int n,
        const float* ra, const float* dec, float ra0, float dec0, float* x,
        float* y, float* z);

/**
 * @brief
 * Equatorial to relative 3D direction cosines (double precision).
 *
 * @details
 * This function computes the x,y,z direction cosines of the specified points
 * relative to the reference point.
 *
 * Note that the z-positions are given by sqrt(1 - x*x - y*y) - 1.
 *
 * @param[in]  n     The number of points.
 * @param[in]  ra    Input position Right Ascensions in radians.
 * @param[in]  dec   Input position Declinations in radians.
 * @param[in]  ra0   Right Ascension of the reference point in radians.
 * @param[in]  dec0  Declination of the reference point in radians.
 * @param[out] x     x-direction-cosines relative to the reference point.
 * @param[out] y     y-direction-cosines relative to the reference point.
 * @param[out] z     z-direction-cosines relative to the reference point.
 */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_tangent_plane_direction_cuda_d(int n,
        const double* ra, const double* dec, double ra0, double dec0,
        double* x, double* y, double* z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_APPARENT_RA_DEC_TO_TANGENT_PLANE_DIRECTION_CUDA_H_ */
