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

#ifndef OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_CUDA_H_
#define OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_CUDA_H_

/**
 * @file oskar_convert_relative_directions_to_enu_directions_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts from relative direction cosines to horizon ENU direction cosines
 * (single precision, GPU version).
 *
 * @details
 * This function transforms the given \f$(l, m, n)\f$ directions in the
 * equatorial frame to \f$(x, y, z)\f$ directions in the horizontal frame.
 *
 * It is equivalent to the product of matrix transformations as follows:
 *
   \f[
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}

        = R_x(\phi) \cdot R_y(-H_0) \cdot R_x(-\delta_0) \cdot

    \begin{bmatrix}
    l \\
    m \\
    n
    \end{bmatrix}
   \f]
 *
 * Here, \f$ R_x \f$ and \f$ R_y \f$ correspond to rotations around
 * the \f$x\f$- and \f$y\f$-axes, respectively.
 * The angles \f$ \phi \f$, \f$ H_0 \f$ and \f$ \delta_0 \f$ correspond to
 * the observer's geodetic latitude, the hour angle and the declination of
 * the phase centre.
 *
 * @param[out] x          ENU direction cosines (East).
 * @param[out] y          ENU direction cosines (North).
 * @param[out] z          ENU direction cosines (up).
 * @param[in]  num_points Number of points to convert.
 * @param[in]  l          Relative direction cosines.
 * @param[in]  m          Relative direction cosines.
 * @param[in]  n          Relative direction cosines.
 * @param[in]  ha0        Hour angle of the origin of the relative directions,
 *                        in radians.
 * @param[in]  dec0       Declination of the origin of the relative directions,
 *                        in radians.
 * @param[in]  lat        Latitude of the ENU coordinate frame, in radians.
 */
OSKAR_EXPORT
void oskar_convert_relative_directions_to_enu_directions_cuda_f(
        float* x, float* y, float* z, int num_points, const float* l,
        const float* m, const float* n, float ha0, float dec0, float lat);

/**
 * @brief
 * Converts from relative direction cosines to horizon ENU direction cosines
 * (double precision, GPU version).
 *
 * @details
 * This function transforms the given \f$(l, m, n)\f$ directions in the
 * equatorial frame to \f$(x, y, z)\f$ directions in the horizontal frame.
 *
 * It is equivalent to the product of matrix transformations as follows:
 *
   \f[
    \begin{bmatrix}
    x \\
    y \\
    z
    \end{bmatrix}

        = R_x(\phi) \cdot R_y(-H_0) \cdot R_x(-\delta_0) \cdot

    \begin{bmatrix}
    l \\
    m \\
    n
    \end{bmatrix}
   \f]
 *
 * Here, \f$ R_x \f$ and \f$ R_y \f$ correspond to rotations around
 * the \f$x\f$- and \f$y\f$-axes, respectively.
 * The angles \f$ \phi \f$, \f$ H_0 \f$ and \f$ \delta_0 \f$ correspond to
 * the observer's geodetic latitude, the hour angle and the declination of
 * the phase centre.
 *
 * @param[out] x          ENU direction cosines (East).
 * @param[out] y          ENU direction cosines (North).
 * @param[out] z          ENU direction cosines (up).
 * @param[in]  num_points Number of points to convert.
 * @param[in]  l          Relative direction cosines.
 * @param[in]  m          Relative direction cosines.
 * @param[in]  n          Relative direction cosines.
 * @param[in]  ha0        Hour angle of the origin of the relative directions,
 *                        in radians.
 * @param[in]  dec0       Declination of the origin of the relative directions,
 *                        in radians.
 * @param[in]  lat        Latitude of the ENU coordinate frame, in radians.
 */
OSKAR_EXPORT
void oskar_convert_relative_directions_to_enu_directions_cuda_d(
        double* x, double* y, double* z, int num_points, const double* l,
        const double* m, const double* n, double ha0, double dec0, double lat);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_convert_relative_directions_to_enu_directions_cudak_f(
        float* __restrict__ x, float* __restrict__ y, float* __restrict__ z,
        const int num_points, const float* __restrict__ l,
        const float* __restrict__ m, const float* __restrict__ n,
        const float cos_ha0, const float sin_ha0, const float cos_dec0,
        const float sin_dec0, const float cos_lat, const float sin_lat);

__global__
void oskar_convert_relative_directions_to_enu_directions_cudak_d(
        double* __restrict__ x, double* __restrict__ y, double* __restrict__ z,
        const int num_points, const double* __restrict__ l,
        const double* __restrict__ m, const double* __restrict__ n,
        const double cos_ha0, const double sin_ha0, const double cos_dec0,
        const double sin_dec0, const double cos_lat, const double sin_lat);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_CUDA_H_ */
