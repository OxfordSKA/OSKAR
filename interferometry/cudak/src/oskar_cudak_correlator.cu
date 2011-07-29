/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/cudak/oskar_cudak_correlator.h"

// Single precision.

#define ONE_OVER_2C 1.66782047599076024788E-9   // 1 / (2c)
#define ONE_OVER_2Cf 1.66782047599076024788E-9f // 1 / (2c)

extern __shared__ float2 smem[];

__device__ float sincf(float a)
{
    return (a == 0.0f) ? 1.0f : sinf(a) / a;
}

__global__
void oskar_cudakf_correlator(const int ns, const int na,
        const float2* k, const float* u, const float* v, const float* l,
        const float* m, const float lambda_bandwidth, float2* vis)
{
    // Determine which index into the visibility matrix this thread block
    // is generating.
    int ai = blockIdx.x; // Column index.
    int aj = blockIdx.y; // Row index.

    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (aj >= ai) return;

    // Determine UV-distance for baseline (common per thread block).
    __device__ __shared__ float uu, vv;
    if (threadIdx.x == 0)
    {
        uu = ONE_OVER_2Cf * lambda_bandwidth * (u[ai] - u[aj]);
        vv = ONE_OVER_2Cf * lambda_bandwidth * (v[ai] - v[aj]);
    }
    __syncthreads();

    // Get pointers to both source vectors for station i and j.
    const float2* sti = &k[ns * ai];
    const float2* stj = &k[ns * aj];

    // Initialise shared memory.
    smem[threadIdx.x] = make_float2(0.0f, 0.0f);
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
        float rb = sincf(uu * l[t] + vv * m[t]);
        float2 temp;
        float2 stit = sti[t];
        float2 stjt = stj[t];

        // Complex-conjugate multiply.
        temp.x = stit.x * stjt.x + stit.y * stjt.y;
        temp.y = stit.x * stjt.y - stit.y * stjt.x;
        smem[threadIdx.x].x += temp.x * rb;
        smem[threadIdx.x].y += temp.y * rb;
    }
    __syncthreads();

    // Accumulate (could optimise this).
    if (threadIdx.x == 0)
    {
        float2 temp = make_float2(0.0f, 0.0f);
        for (int i = 0; i < blockDim.x; ++i)
        {
            temp.x += smem[i].x;
            temp.y += smem[i].y;
        }

        // Determine 1D index.
        int idx = aj*(na-1) - (aj-1)*aj/2 + ai - aj - 1;

        // Modify existing visibility.
        vis[idx].x += temp.x;
        vis[idx].y += temp.y;
    }
}

// Double precision.

extern __shared__ double2 smemd[];

__device__ double sinc(double a)
{
    return (a == 0.0) ? 1.0 : sin(a) / a;
}

__global__
void oskar_cudakd_correlator(const int ns, const int na,
        const double2* k, const double* u, const double* v, const double* l,
        const double* m, const double lambda_bandwidth, double2* vis)
{
    // Determine which index into the visibility matrix this thread block
    // is generating.
    int ai = blockIdx.x; // Column index.
    int aj = blockIdx.y; // Row index.

    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (aj >= ai) return;

    // Determine UV-distance for baseline (common per thread block).
    __device__ __shared__ float uu, vv;
    if (threadIdx.x == 0)
    {
        uu = ONE_OVER_2C * lambda_bandwidth * (u[ai] - u[aj]);
        vv = ONE_OVER_2C * lambda_bandwidth * (v[ai] - v[aj]);
    }
    __syncthreads();

    // Get pointers to both source vectors for station i and j.
    const double2* sti = &k[ns * ai];
    const double2* stj = &k[ns * aj];

    // Initialise shared memory.
    smemd[threadIdx.x] = make_double2(0.0, 0.0);
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
        double rb = sinc(uu * l[t] + vv * m[t]);
        double2 temp;
        double2 stit = sti[t];
        double2 stjt = stj[t];

        // Complex-conjugate multiply.
        temp.x = stit.x * stjt.x + stit.y * stjt.y;
        temp.y = stit.x * stjt.y - stit.y * stjt.x;
        smemd[threadIdx.x].x += temp.x * rb;
        smemd[threadIdx.x].y += temp.y * rb;
    }
    __syncthreads();

    // Accumulate (could optimise this).
    if (threadIdx.x == 0)
    {
        double2 temp = make_double2(0.0, 0.0);
        for (int i = 0; i < blockDim.x; ++i)
        {
            temp.x += smemd[i].x;
            temp.y += smemd[i].y;
        }

        // Determine 1D index.
        int idx = aj*(na-1) - (aj-1)*aj/2 + ai - aj - 1;

        // Modify existing visibility.
        vis[idx].x += temp.x;
        vis[idx].y += temp.y;
    }
}
