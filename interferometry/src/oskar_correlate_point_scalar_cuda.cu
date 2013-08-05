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

#include "interferometry/oskar_correlate_point_scalar_cuda.h"
#include "math/oskar_multiply_inline.h"
#include "math/oskar_sinc.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_correlate_point_scalar_cuda_f(int num_sources,
        int num_stations, const float2* d_jones,
        const float* d_source_I, const float* d_source_l,
        const float* d_source_m, const float* d_station_u,
        const float* d_station_v, float frac_bandwidth, float2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float2);
    oskar_correlate_point_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_l, d_source_m,
            d_station_u, d_station_v, frac_bandwidth, d_vis);
}

/* Double precision. */
void oskar_correlate_point_scalar_cuda_d(int num_sources,
        int num_stations, const double2* d_jones,
        const double* d_source_I, const double* d_source_l,
        const double* d_source_m, const double* d_station_u,
        const double* d_station_v, double frac_bandwidth, double2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(num_stations, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double2);
    oskar_correlate_point_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_l, d_source_m,
            d_station_u, d_station_v, frac_bandwidth, d_vis);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Indices into the visibility/baseline matrix. */
#define AI blockIdx.x /* Column index. */
#define AJ blockIdx.y /* Row index. */

extern __shared__ float2  smem_f[];
extern __shared__ double2 smem_d[];

/* Single precision. */
__global__
void oskar_correlate_point_scalar_cudak_f(const int num_sources,
        const int num_stations, const float2* jones, const float* source_I,
        const float* source_l, const float* source_m,
        const float* station_u, const float* station_v,
        const float frac_bandwidth, float2* vis)
{
    /* Return immediately if in the wrong half of the visibility matrix. */
    if (AJ >= AI) return;

    /* Common values per thread block. */
    __shared__ float uu, vv;
    if (threadIdx.x == 0)
    {
        /* Determine UV-distance for baseline (common per thread block). */
        uu = 0.5f * frac_bandwidth * (station_u[AI] - station_u[AJ]);
        vv = 0.5f * frac_bandwidth * (station_v[AI] - station_v[AJ]);
    }
    __syncthreads();

    /* Get pointers to both source vectors for station i and j. */
    const float2* station_i = &jones[num_sources * AI];
    const float2* station_j = &jones[num_sources * AJ];

    /* Each thread loops over a subset of the sources. */
    {
        float2 sum = make_float2(0.0f, 0.0f); /* Partial sum per thread. */
        for (int t = threadIdx.x; t < num_sources; t += blockDim.x)
        {
            /* Compute bandwidth-smearing term first (register optimisation). */
            float rb = oskar_sinc_f(uu * source_l[t] + vv * source_m[t]);
            float2 c_a = station_i[t];
            float2 c_b = station_j[t];
            float2 temp;

            /* Complex-conjugate multiply. */
            oskar_multiply_complex_conjugate_f(&temp, &c_a, &c_b);

            /* Multiply by the source brightness. */
            temp.x *= source_I[t];
            temp.y *= source_I[t];

            /* Multiply result by bandwidth-smearing term. */
            sum.x += temp.x * rb;
            sum.y += temp.y * rb;
        }
        smem_f[threadIdx.x] = sum;
    }
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        float2 sum = make_float2(0.0f, 0.0f);
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.x += smem_f[i].x;
            sum.y += smem_f[i].y;
        }

        /* Determine 1D index. */
        int idx = AJ*(num_stations-1) - (AJ-1)*AJ/2 + AI - AJ - 1;

        /* Modify existing visibility. */
        vis[idx].x += sum.x;
        vis[idx].y += sum.y;
    }
}

/* Double precision. */
__global__
void oskar_correlate_point_scalar_cudak_d(const int num_sources,
        const int num_stations, const double2* jones, const double* source_I,
        const double* source_l, const double* source_m,
        const double* station_u, const double* station_v,
        const double frac_bandwidth, double2* vis)
{
    /* Return immediately if in the wrong half of the visibility matrix. */
    if (AJ >= AI) return;

    /* Common values per thread block. */
    __shared__ double uu, vv;
    if (threadIdx.x == 0)
    {
        /* Determine UV-distance for baseline (common per thread block). */
        uu = 0.5 * frac_bandwidth * (station_u[AI] - station_u[AJ]);
        vv = 0.5 * frac_bandwidth * (station_v[AI] - station_v[AJ]);
    }
    __syncthreads();

    /* Get pointers to both source vectors for station i and j. */
    const double2* station_i = &jones[num_sources * AI];
    const double2* station_j = &jones[num_sources * AJ];

    /* Each thread loops over a subset of the sources. */
    {
        double2 sum = make_double2(0.0, 0.0); /* Partial sum per thread. */
        for (int t = threadIdx.x; t < num_sources; t += blockDim.x)
        {
            /* Compute bandwidth-smearing term first (register optimisation). */
            double rb = oskar_sinc_d(uu * source_l[t] + vv * source_m[t]);
            double2 c_a = station_i[t];
            double2 c_b = station_j[t];
            double2 temp;

            /* Complex-conjugate multiply. */
            oskar_multiply_complex_conjugate_d(&temp, &c_a, &c_b);

            /* Multiply by the source brightness. */
            temp.x *= source_I[t];
            temp.y *= source_I[t];

            /* Multiply result by bandwidth-smearing term. */
            sum.x += temp.x * rb;
            sum.y += temp.y * rb;
        }
        smem_d[threadIdx.x] = sum;
    }
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources for this baseline. */
        double2 sum = make_double2(0.0, 0.0);
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.x += smem_d[i].x;
            sum.y += smem_d[i].y;
        }

        /* Determine 1D index. */
        int idx = AJ*(num_stations-1) - (AJ-1)*AJ/2 + AI - AJ - 1;

        /* Modify existing visibility. */
        vis[idx].x += sum.x;
        vis[idx].y += sum.y;
    }
}
