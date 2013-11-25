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

#include <oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cuda_f(
        float* x, float* y, float* z, int num_points, const float* l,
        const float* m, const float* n, float ha0, float dec0, float lat)
{
    float sin_ha0, cos_ha0, sin_dec0, cos_dec0, sin_lat, cos_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_points + num_threads - 1) / num_threads;

    /* Compute sines and cosines of Euler angles and call kernel to perform
     * the transformation. */
    sin_ha0  = (float) sin(ha0);
    cos_ha0  = (float) cos(ha0);
    sin_dec0 = (float) sin(dec0);
    cos_dec0 = (float) cos(dec0);
    sin_lat  = (float) sin(lat);
    cos_lat  = (float) cos(lat);
    oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (x, y, z, num_points, l, m, n,
            cos_ha0, sin_ha0, cos_dec0, sin_dec0, cos_lat, sin_lat);
}

void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cuda_d(
        double* x, double* y, double* z, int num_points, const double* l,
        const double* m, const double* n, double ha0, double dec0, double lat)
{
    double sin_ha0, cos_ha0, sin_dec0, cos_dec0, sin_lat, cos_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_points + num_threads - 1) / num_threads;

    /* Compute sines and cosines of Euler angles and call kernel to perform
     * the transformation. */
    sin_ha0  = sin(ha0);
    cos_ha0  = cos(ha0);
    sin_dec0 = sin(dec0);
    cos_dec0 = cos(dec0);
    sin_lat  = sin(lat);
    cos_lat  = cos(lat);
    oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (x, y, z, num_points, l, m, n,
            cos_ha0, sin_ha0, cos_dec0, sin_dec0, cos_lat, sin_lat);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cudak_f(
        float* x, float* y, float* z, int num_points, const float* l,
        const float* m, const float* n, float cos_ha0, float sin_ha0,
        float cos_dec0, float sin_dec0, float cos_lat, float sin_lat)
{
    float l_, m_, n_, x_, y_, z_, t;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    l_ = l[i];
    m_ = m[i];
    n_ = n[i];
    x_ = l_ * cos_ha0 + m_ * sin_ha0 * sin_dec0 - n_ * sin_ha0 * cos_dec0;
    t = sin_lat * cos_ha0;
    y_ = -l_ * sin_lat * sin_ha0 +
            m_ * (cos_lat * cos_dec0 + t * sin_dec0) +
            n_ * (cos_lat * sin_dec0 - t * cos_dec0);
    t = cos_lat * cos_ha0;
    z_ = l_ * cos_lat * sin_ha0 +
            m_ * (sin_lat * cos_dec0 - t * sin_dec0) +
            n_ * (sin_lat * sin_dec0 + t * cos_dec0);
    x[i] = x_;
    y[i] = y_;
    z[i] = z_;
}

/* Double precision. */
__global__
void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_cudak_d(
        double* x, double* y, double* z, int num_points, const double* l,
        const double* m, const double* n, double cos_ha0, double sin_ha0,
        double cos_dec0, double sin_dec0, double cos_lat, double sin_lat)
{
    double l_, m_, n_, x_, y_, z_, t;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    l_ = l[i];
    m_ = m[i];
    n_ = n[i];
    x_ = l_ * cos_ha0 + m_ * sin_ha0 * sin_dec0 - n_ * sin_ha0 * cos_dec0;
    t = sin_lat * cos_ha0;
    y_ = -l_ * sin_lat * sin_ha0 +
            m_ * (cos_lat * cos_dec0 + t * sin_dec0) +
            n_ * (cos_lat * sin_dec0 - t * cos_dec0);
    t = cos_lat * cos_ha0;
    z_ = l_ * cos_lat * sin_ha0 +
            m_ * (sin_lat * cos_dec0 - t * sin_dec0) +
            n_ * (sin_lat * sin_dec0 + t * cos_dec0);
    x[i] = x_;
    y[i] = y_;
    z[i] = z_;
}
