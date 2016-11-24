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

#include "convert/oskar_convert_ecef_to_station_uvw_cuda.h"
#include "convert/private_convert_ecef_to_station_uvw_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_ecef_to_station_uvw_cuda_f(int num_stations,
        const float* d_x, const float* d_y, const float* d_z,
        float ha0_rad, float dec0_rad, float* d_u, float* d_v, float* d_w)
{
    float sin_ha0, cos_ha0, sin_dec0, cos_dec0;

    /* Define block and grid sizes. */
    int num_blocks, num_threads = 256;
    num_blocks = (num_stations + num_threads - 1) / num_threads;

    /* Precompute trig. */
    sin_ha0  = (float)sin(ha0_rad);
    cos_ha0  = (float)cos(ha0_rad);
    sin_dec0 = (float)sin(dec0_rad);
    cos_dec0 = (float)cos(dec0_rad);

    oskar_convert_ecef_to_station_uvw_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_stations, d_x, d_y, d_z, sin_ha0, cos_ha0, sin_dec0, cos_dec0,
                d_u, d_v, d_w);
}

/* Double precision. */
void oskar_convert_ecef_to_station_uvw_cuda_d(int num_stations,
        const double* d_x, const double* d_y, const double* d_z,
        double ha0_rad, double dec0_rad, double* d_u, double* d_v, double* d_w)
{
    double sin_ha0, cos_ha0, sin_dec0, cos_dec0;

    /* Define block and grid sizes. */
    int num_blocks, num_threads = 256;
    num_blocks = (num_stations + num_threads - 1) / num_threads;

    /* Precompute trig. */
    sin_ha0  = sin(ha0_rad);
    cos_ha0  = cos(ha0_rad);
    sin_dec0 = sin(dec0_rad);
    cos_dec0 = cos(dec0_rad);

    oskar_convert_ecef_to_station_uvw_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_stations, d_x, d_y, d_z, sin_ha0, cos_ha0, sin_dec0, cos_dec0,
                d_u, d_v, d_w);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_ecef_to_station_uvw_cudak_f(const int num_stations,
        const float* restrict x, const float* restrict y,
        const float* restrict z, const float sin_ha0,
        const float cos_ha0, const float sin_dec0, const float cos_dec0,
        float* restrict u, float* restrict v, float* restrict w)
{
    /* Get station ID. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_stations) return;

    oskar_convert_ecef_to_station_uvw_inline_f(x[i], y[i], z[i],
            sin_ha0, cos_ha0, sin_dec0, cos_dec0, &u[i], &v[i], &w[i]);
}

/* Double precision. */
__global__
void oskar_convert_ecef_to_station_uvw_cudak_d(const int num_stations,
        const double* restrict x, const double* restrict y,
        const double* restrict z, const double sin_ha0,
        const double cos_ha0, const double sin_dec0, const double cos_dec0,
        double* restrict u, double* restrict v, double* restrict w)
{
    /* Get station ID. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_stations) return;

    oskar_convert_ecef_to_station_uvw_inline_d(x[i], y[i], z[i],
            sin_ha0, cos_ha0, sin_dec0, cos_dec0, &u[i], &v[i], &w[i]);
}

#ifdef __cplusplus
}
#endif
