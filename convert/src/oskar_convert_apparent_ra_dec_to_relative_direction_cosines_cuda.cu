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

#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda.h>
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_f(
        int num_points, const float* d_ra, const float* d_dec, float ra0,
        float dec0, float* d_l, float* d_m, float* d_n)
{
    float cos_dec0, sin_dec0;
    int num_blocks, num_threads = 256;

    /* Compute direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cos_dec0 = (float) cos(dec0);
    sin_dec0 = (float) sin(dec0);

    oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, d_ra, d_dec, ra0, cos_dec0, sin_dec0, d_l, d_m, d_n);
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cuda_d(
        int num_points, const double* d_ra, const double* d_dec, double ra0,
        double dec0, double* d_l, double* d_m, double* d_n)
{
    double cos_dec0, sin_dec0;
    int num_blocks, num_threads = 256;

    /* Compute direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cos_dec0 = cos(dec0);
    sin_dec0 = sin(dec0);

    oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, d_ra, d_dec, ra0, cos_dec0, sin_dec0, d_l, d_m, d_n);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_f(
        const int num_points, const float* __restrict__ ra,
        const float* __restrict__ dec, const float ra0, const float cos_dec0,
        const float sin_dec0, float* __restrict__ l, float* __restrict__ m,
        float* __restrict__ n)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_f(
            ra[i], dec[i], ra0, cos_dec0, sin_dec0, &l[i], &m[i], &n[i]);
}

/* Double precision. */
__global__
void oskar_convert_apparent_ra_dec_to_relative_direction_cosines_cudak_d(
        const int num_points, const double* __restrict__ ra,
        const double* __restrict__ dec, const double ra0, const double cos_dec0,
        const double sin_dec0, double* __restrict__ l, double* __restrict__ m,
        double* __restrict__ n)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_apparent_ra_dec_to_relative_direction_cosines_inline_d(
            ra[i], dec[i], ra0, cos_dec0, sin_dec0, &l[i], &m[i], &n[i]);
}

#ifdef __cplusplus
}
#endif
