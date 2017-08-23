/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "interferometer/oskar_evaluate_jones_K_cuda.h"

/* Kernels. ================================================================ */

#define BLK_STATION 2
#define BLK_SOURCE 128

/* Single precision. */
__global__
void oskar_evaluate_jones_K_cudak_f(float2* restrict jones,
        const int num_sources, const float* restrict l,
        const float* restrict m, const float* restrict n,
        const int num_stations, const float* restrict u,
        const float* restrict v, const float* restrict w,
        const float wavenumber, const float* restrict source_filter,
        const float source_filter_min, const float source_filter_max)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; /* Source index. */
    const int a = blockDim.y * blockIdx.y + threadIdx.y; /* Station index. */

    /* Cache source and station data from global memory. */
    __shared__ float l_[BLK_SOURCE], m_[BLK_SOURCE], n_[BLK_SOURCE];
    __shared__ float f_[BLK_SOURCE];
    __shared__ float u_[BLK_STATION], v_[BLK_STATION], w_[BLK_STATION];
    if (s < num_sources && threadIdx.y == 0)
    {
        l_[threadIdx.x] = l[s];
        m_[threadIdx.x] = m[s];
        n_[threadIdx.x] = n[s] - 1.0f;
        f_[threadIdx.x] = source_filter[s];
    }
    if (a < num_stations && threadIdx.x == 0)
    {
        u_[threadIdx.y] = wavenumber * u[a];
        v_[threadIdx.y] = wavenumber * v[a];
        w_[threadIdx.y] = wavenumber * w[a];
    }
    __syncthreads();

    /* Compute the geometric phase of the source direction. */
    float2 weight = make_float2(0.0f, 0.0f);
    if (f_[threadIdx.x] > source_filter_min &&
            f_[threadIdx.x] <= source_filter_max)
    {
        float phase;
        phase =  u_[threadIdx.y] * l_[threadIdx.x];
        phase += v_[threadIdx.y] * m_[threadIdx.x];
        phase += w_[threadIdx.y] * n_[threadIdx.x];
        sincosf(phase, &weight.y, &weight.x);
    }

    /* Write result to global memory. */
    if (s < num_sources && a < num_stations)
        jones[s + num_sources * a] = weight;
}

/* Double precision. */
__global__
void oskar_evaluate_jones_K_cudak_d(double2* restrict jones,
        const int num_sources, const double* restrict l,
        const double* restrict m, const double* restrict n,
        const int num_stations, const double* restrict u,
        const double* restrict v, const double* restrict w,
        const double wavenumber, const double* restrict source_filter,
        const double source_filter_min, const double source_filter_max)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; /* Source index. */
    const int a = blockDim.y * blockIdx.y + threadIdx.y; /* Station index. */

    /* Cache source and station data from global memory. */
    __shared__ double l_[BLK_SOURCE], m_[BLK_SOURCE], n_[BLK_SOURCE];
    __shared__ double f_[BLK_SOURCE];
    __shared__ double u_[BLK_STATION], v_[BLK_STATION], w_[BLK_STATION];
    if (s < num_sources && threadIdx.y == 0)
    {
        l_[threadIdx.x] = l[s];
        m_[threadIdx.x] = m[s];
        n_[threadIdx.x] = n[s] - 1.0;
        f_[threadIdx.x] = source_filter[s];
    }
    if (a < num_stations && threadIdx.x == 0)
    {
        u_[threadIdx.y] = wavenumber * u[a];
        v_[threadIdx.y] = wavenumber * v[a];
        w_[threadIdx.y] = wavenumber * w[a];
    }
    __syncthreads();

    /* Compute the geometric phase of the source direction. */
    double2 weight = make_double2(0.0, 0.0);
    if (f_[threadIdx.x] > source_filter_min &&
            f_[threadIdx.x] <= source_filter_max)
    {
        double phase;
        phase =  u_[threadIdx.y] * l_[threadIdx.x];
        phase += v_[threadIdx.y] * m_[threadIdx.x];
        phase += w_[threadIdx.y] * n_[threadIdx.x];
        sincos(phase, &weight.y, &weight.x);
    }

    /* Write result to global memory. */
    if (s < num_sources && a < num_stations)
        jones[s + num_sources * a] = weight;
}


#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_jones_K_cuda_f(float2* d_jones, int num_sources,
        const float* d_l, const float* d_m, const float* d_n,
        int num_stations, const float* d_u, const float* d_v,
        const float* d_w, float wavenumber, const float* d_source_filter,
        float source_filter_min, float source_filter_max)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(BLK_SOURCE, BLK_STATION);
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);

    /* Compute DFT phase weights for K. */
    oskar_evaluate_jones_K_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (d_jones, num_sources, d_l, d_m, d_n, num_stations, d_u, d_v, d_w,
            wavenumber, d_source_filter, source_filter_min, source_filter_max);
}

/* Double precision. */
void oskar_evaluate_jones_K_cuda_d(double2* d_jones, int num_sources,
        const double* d_l, const double* d_m, const double* d_n,
        int num_stations, const double* d_u, const double* d_v,
        const double* d_w, double wavenumber, const double* d_source_filter,
        double source_filter_min, double source_filter_max)
{
    /* Define block and grid sizes. */
    const dim3 num_threads(BLK_SOURCE, BLK_STATION);
    const dim3 num_blocks((num_sources + num_threads.x - 1) / num_threads.x,
            (num_stations + num_threads.y - 1) / num_threads.y);

    /* Compute DFT phase weights for K. */
    oskar_evaluate_jones_K_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (d_jones, num_sources, d_l, d_m, d_n, num_stations, d_u, d_v, d_w,
            wavenumber, d_source_filter, source_filter_min, source_filter_max);
}

#ifdef __cplusplus
}
#endif
