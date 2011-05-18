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

#include "cuda/kernels/oskar_cudak_correlator.h"

// Single precision.

extern __shared__ float2 smem[];

__device__ float sincf(float a)
{
    if (a == 0.0f) return 1.0f;
    else return sinf(a) / a;
//    float b = a + 1.0e-7;
//    return sinf(b) / b;
}

__global__
void oskar_cudakf_correlator(int ns, int na, const float2* k,
        const float* lmdist, const float* uvdist, float2* vis)
{
    // Determine which index into the visibility matrix this thread block
    // is generating.
    int ai = blockIdx.x; // Column index.
    int aj = blockIdx.y; // Row index.

    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (aj >= ai) return;

    // Determine 1D index.
    int ajp = aj - 1;
    int idx = aj*(na-1) - ajp*(ajp + 1)/2 + ai - aj - 1;

    // Get pointers to both source vectors for station i and j.
    const float2* sti = &k[ns * ai];
    const float2* stj = &k[ns * aj];

    // Initialise shared memory.
    smem[threadIdx.x] = make_float2(0.0f, 0.0f);
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
        float rb = sincf(lmdist[t] * uvdist[idx]);
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
        float2 temp;
        for (int i = 0; i < blockDim.x; ++i)
        {
            temp.x += smem[i].x;
            temp.y += smem[i].y;
        }
        vis[idx].x += temp.x;
        vis[idx].y += temp.y;
    }
}

// Double precision.

extern __shared__ double2 smemd[];

__device__ double sinc(double a)
{
    if (a == 0.0) return 1.0;
    else return sin(a) / a;
//    double b = a + 2.0e-17;
//    return sin(b) / b;
}

__global__
void oskar_cudakd_correlator(int ns, int na, const double2* k,
        const double* lmdist, const double* uvdist, double2* vis)
{
    // Determine which index into the visibility matrix this thread block
    // is generating.
    int ai = blockIdx.x; // Column index.
    int aj = blockIdx.y; // Row index.

    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (aj >= ai) return;

    // Determine 1D index.
    int ajp = aj - 1;
    int idx = aj*(na-1) - ajp*(ajp + 1)/2 + ai - aj - 1;

    // Get pointers to both source vectors for station i and j.
    const double2* sti = &k[ns * ai];
    const double2* stj = &k[ns * aj];

    // Initialise shared memory.
    smemd[threadIdx.x] = make_double2(0.0, 0.0);
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
        double rb = sincf(lmdist[t] * uvdist[idx]);
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
        double2 temp;
        for (int i = 0; i < blockDim.x; ++i)
        {
            temp.x += smemd[i].x;
            temp.y += smemd[i].y;
        }
        vis[idx].x += temp.x;
        vis[idx].y += temp.y;
    }
}
