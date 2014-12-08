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

#include <oskar_convert_lon_lat_to_relative_directions_cuda.h>
#include <private_convert_lon_lat_to_relative_directions_inline.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_convert_lon_lat_to_relative_directions_cuda_f(int num_points,
        const float* d_lon_rad, const float* d_dec_rad, float lon0_rad,
        float lat0_rad, float* d_l, float* d_m, float* d_n)
{
    float cos_lat0, sin_lat0;
    int num_blocks, num_threads = 256;

    /* Compute direction cosines relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cos_lat0 = (float) cos(lat0_rad);
    sin_lat0 = (float) sin(lat0_rad);
    oskar_convert_lon_lat_to_relative_directions_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, d_lon_rad, d_dec_rad, lon0_rad, cos_lat0, sin_lat0,
                d_l, d_m, d_n);
}

/* Double precision. */
void oskar_convert_lon_lat_to_relative_directions_cuda_d(int num_points,
        const double* d_lon_rad, const double* d_lat_rad, double lon0_rad,
        double lat0_rad, double* d_l, double* d_m, double* d_n)
{
    double cos_dec0, sin_dec0;
    int num_blocks, num_threads = 256;

    /* Compute direction cosines relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cos_dec0 = cos(lat0_rad);
    sin_dec0 = sin(lat0_rad);
    oskar_convert_lon_lat_to_relative_directions_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, d_lon_rad, d_lat_rad, lon0_rad, cos_dec0, sin_dec0,
                d_l, d_m, d_n);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_lon_lat_to_relative_directions_cudak_f(const int num_points,
        const float* restrict lon_rad, const float* restrict lat_rad,
        const float lon0_rad, const float cos_lat0, const float sin_lat0,
        float* restrict l, float* restrict m, float* restrict n)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_lon_lat_to_relative_directions_inline_f(
            lon_rad[i], lat_rad[i], lon0_rad, cos_lat0, sin_lat0,
            &l[i], &m[i], &n[i]);
}

/* Double precision. */
__global__
void oskar_convert_lon_lat_to_relative_directions_cudak_d(const int num_points,
        const double* restrict lon_rad, const double* restrict lat_rad,
        const double lon0_rad, const double cos_lat0, const double sin_lat0,
        double* restrict l, double* restrict m, double* restrict n)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_lon_lat_to_relative_directions_inline_d(
            lon_rad[i], lat_rad[i], lon0_rad, cos_lat0, sin_lat0,
            &l[i], &m[i], &n[i]);
}

#ifdef __cplusplus
}
#endif
