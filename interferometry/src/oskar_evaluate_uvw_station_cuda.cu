/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_evaluate_uvw_station_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_uvw_station_cuda_f(float* d_u, float* d_v, float* d_w,
        int num_stations, const float* d_x, const float* d_y,
        const float* d_z, float ha0_rad, float dec0_rad)
{
    /* Define block and grid sizes. */
    const int num_threads = 256;
    const int num_blocks = (num_stations + num_threads - 1) / num_threads;

    /* Call the CUDA kernel. */
    oskar_evaluate_uvw_station_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (d_u, d_v, d_w, num_stations, d_x, d_y, d_z, ha0_rad, dec0_rad);
}

/* Double precision. */
void oskar_evaluate_uvw_station_cuda_d(double* d_u, double* d_v, double* d_w,
        int num_stations, const double* d_x, const double* d_y,
        const double* d_z, double ha0, double dec0)
{
    /* Define block and grid sizes. */
    const int num_threads = 256;
    const int num_blocks = (num_stations + num_threads - 1) / num_threads;

    /* Call the CUDA kernel. */
    oskar_evaluate_uvw_station_cudak_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (d_u, d_v, d_w, num_stations, d_x, d_y, d_z, ha0, dec0);
}

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_uvw_station_cudak_f(float* u, float* v, float* w,
        int num_stations, const float* x, const float* y, const float* z,
        float ha0_rad, float dec0_rad)
{
    /* Pre-compute sine and cosine of input angles. */
    __device__ __shared__ float sinHa0, cosHa0, sinDec0, cosDec0;
    if (threadIdx.x == 0)
    {
        sincosf(ha0_rad, &sinHa0, &cosHa0);
        sincosf(dec0_rad, &sinDec0, &cosDec0);
    }
    __syncthreads();

    /* Get station ID. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_stations) return;

    /* Cache input coordinates. */
    float cx = x[i], cy = y[i], cz = z[i], cv, cw, t;

    /* Do the rotation. */
    t = cx * cosHa0;
    t -= cy * sinHa0;
    cv = cz * cosDec0;
    cv -= sinDec0 * t;
    cw = cosDec0 * t;
    cw += cz * sinDec0;
    t =  cx * sinHa0;
    t += cy * cosHa0;
    u[i] = t;
    v[i] = cv;
    w[i] = cw;
}

/* Double precision. */
__global__
void oskar_evaluate_uvw_station_cudak_d(double* u, double* v, double* w,
        int num_stations, const double* x, const double* y, const double* z,
        double ha0_rad, double dec0_rad)
{
    /* Pre-compute sine and cosine of input angles. */
    __device__ __shared__ double sinHa0, cosHa0, sinDec0, cosDec0;
    if (threadIdx.x == 0)
    {
        sincos(ha0_rad, &sinHa0, &cosHa0);
        sincos(dec0_rad, &sinDec0, &cosDec0);
    }
    __syncthreads();

    /* Get station ID. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_stations) return;

    /* Cache input coordinates. */
    double cx = x[i], cy = y[i], cz = z[i], cv, cw, t;

    /* Do the rotation. */
    t = cx * cosHa0;
    t -= cy * sinHa0;
    cv = cz * cosDec0;
    cv -= sinDec0 * t;
    cw = cosDec0 * t;
    cw += cz * sinDec0;
    t =  cx * sinHa0;
    t += cy * cosHa0;
    u[i] = t;
    v[i] = cv;
    w[i] = cw;
}

#ifdef __cplusplus
}
#endif
