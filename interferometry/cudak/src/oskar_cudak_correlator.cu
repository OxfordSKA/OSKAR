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
#include "math/cudak/oskar_cudaf_mul_mat2c_mat2c.h"
#include "math/cudak/oskar_cudaf_mul_mat2c_mat2h.h"
#include "math/cudak/oskar_cudaf_mul_mat2c_mat2c_conj_trans_to_hermitian.h"

// Single precision.

#define ONE_OVER_2C 1.66782047599076024788E-9   // 1 / (2c)
#define ONE_OVER_2Cf 1.66782047599076024788E-9f // 1 / (2c)

extern __shared__ float4c smem[];

__device__ __forceinline__ float sincf(float a)
{
    return (a == 0.0f) ? 1.0f : sinf(a) / a;
}

#define AI blockIdx.x // Column index.
#define AJ blockIdx.y // Row index.

__global__
void oskar_cudak_correlator_f(const int ns, const int na,
        const float4c* jones, const float* source_I, const float* source_Q,
        const float* source_U, const float* source_V, const float* u,
        const float* v, const float* l, const float* m,
        const float lambda_bandwidth, float4c* vis)
{
    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (AJ >= AI) return;

    // Common things per thread block.
    __shared__ float uu, vv;
    __shared__ const float4c *sti, *stj;
    if (threadIdx.x == 0)
    {
        // Determine UV-distance for baseline (common per thread block).
        uu = ONE_OVER_2Cf * lambda_bandwidth * (u[AI] - u[AJ]);
        vv = ONE_OVER_2Cf * lambda_bandwidth * (v[AI] - v[AJ]);

        // Get pointers to both source vectors for station i and j.
        sti = &jones[ns * AI];
        stj = &jones[ns * AJ];
    }
    __syncthreads();

    // Initialise shared memory.
    {
        smem[threadIdx.x].a = make_float2(0.0f, 0.0f);
        smem[threadIdx.x].b = make_float2(0.0f, 0.0f);
        smem[threadIdx.x].c = make_float2(0.0f, 0.0f);
        smem[threadIdx.x].d = make_float2(0.0f, 0.0f);
    }

    // Each thread loops over a subset of the sources.
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
        // Construct source brightness matrix.
        float4 source_B;
        {
            float s_I = source_I[t];
            float s_Q = source_Q[t];
            source_B.x = s_I + s_Q;
            source_B.y = source_U[t];
            source_B.z = source_V[t];
            source_B.w = s_I - s_Q;
        }

        // Multiply first Jones matrix with source coherency matrix.
        float4c c_a = sti[t];
        oskar_cudaf_mul_mat2c_mat2h_f(c_a, source_B);

        // Multiply result with second (Hermitian transposed) Jones matrix.
        float4c c_b = stj[t];
        oskar_cudaf_mul_mat2c_mat2c_conj_trans_to_hermitian_f(c_a, c_b);

        // Multiply result by bandwidth-smearing term.
        float rb = sincf(uu * l[t] + vv * m[t]);
        smem[threadIdx.x].a.x += c_a.a.x * rb;
        smem[threadIdx.x].a.y += c_a.a.y * rb;
        smem[threadIdx.x].b.x += c_a.b.x * rb;
        smem[threadIdx.x].b.y += c_a.b.y * rb;
        smem[threadIdx.x].d.x += c_a.d.x * rb;
        smem[threadIdx.x].d.y += c_a.d.y * rb;
    }
    __syncthreads();

    // Accumulate (could optimise this).
    if (threadIdx.x == 0)
    {
        float4c temp;
        temp.a = make_float2(0.0f, 0.0f);
        temp.b = make_float2(0.0f, 0.0f);
        temp.d = make_float2(0.0f, 0.0f);
        for (int i = 0; i < blockDim.x; ++i)
        {
            temp.a.x += smem[i].a.x;
            temp.a.y += smem[i].a.y;
            temp.b.x += smem[i].b.x;
            temp.b.y += smem[i].b.y;
            temp.d.x += smem[i].d.x;
            temp.d.y += smem[i].d.y;
        }

        // Determine 1D index.
        int idx = AJ*(na-1) - (AJ-1)*AJ/2 + AI - AJ - 1;

        // Modify existing visibility.
        vis[idx].a.x += temp.a.x;
        vis[idx].a.y += temp.a.y;
        vis[idx].b.x += temp.b.x;
        vis[idx].b.y += temp.b.y;
        vis[idx].d.x += temp.d.x;
        vis[idx].d.y += temp.d.y;
    }
}

// Double precision.

extern __shared__ double2 smemd[];

__device__ __forceinline__ double sinc(double a)
{
    return (a == 0.0) ? 1.0 : sin(a) / a;
}

__global__
void oskar_cudak_correlator_d(const int ns, const int na,
        const double4c* jones, const double* source_I, const double* source_Q,
        const double* source_U, const double* source_V, const double* u,
        const double* v, const double* l, const double* m,
        const double lambda_bandwidth, double4c* vis)
{
}
