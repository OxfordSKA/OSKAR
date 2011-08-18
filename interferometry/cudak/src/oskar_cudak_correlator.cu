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

// Single precision.

#define ONE_OVER_2C 1.66782047599076024788E-9   // 1 / (2c)
#define ONE_OVER_2Cf 1.66782047599076024788E-9f // 1 / (2c)

extern __shared__ float4c smem[];

__device__ __forceinline__ float sincf(float a)
{
    return (a == 0.0f) ? 1.0f : sinf(a) / a;
}

__global__
void oskar_cudak_correlator_f(const int ns, const int na,
		const float4c* j1, const float4c* b, const float4c* j2,
		const float* u, const float* v, const float* l, const float* m,
		const float lambda_bandwidth, float4c* vis)
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
    const float4c* sti = &j1[ns * ai];
    const float4c* stj = &j2[ns * aj];

    // Initialise shared memory.
    smem[threadIdx.x].a = make_float2(0.0f, 0.0f);
    smem[threadIdx.x].b = make_float2(0.0f, 0.0f);
    smem[threadIdx.x].c = make_float2(0.0f, 0.0f);
    smem[threadIdx.x].d = make_float2(0.0f, 0.0f);

    // Each thread loops over a subset of the sources.
    for (int t = threadIdx.x; t < ns; t += blockDim.x)
    {
    	// Multiply first Jones matrix with source coherency matrix.
        float4c c_a, c_b, c_c;
        c_a = sti[t];
        c_b = b[t];
        oskar_cudaf_mul_mat2c_mat2c_f(c_a, c_b, c_c);

        // Multiply result with second (Hermitian transposed) Jones matrix.
        c_b = stj[t];
        oskar_cudaf_mul_mat2c_mat2c_f(c_c, c_b, c_a);

        // Multiply result by bandwidth-smearing term.
        float rb = sincf(uu * l[t] + vv * m[t]);
        smem[threadIdx.x].a.x += c_a.a.x * rb;
        smem[threadIdx.x].a.y += c_a.a.y * rb;
        smem[threadIdx.x].b.x += c_a.b.x * rb;
        smem[threadIdx.x].b.y += c_a.b.y * rb;
        smem[threadIdx.x].c.x += c_a.c.x * rb;
        smem[threadIdx.x].c.y += c_a.c.y * rb;
        smem[threadIdx.x].d.x += c_a.d.x * rb;
        smem[threadIdx.x].d.y += c_a.d.y * rb;
    }
    __syncthreads();

    // TODO Accumulate.
}

// Double precision.

extern __shared__ double2 smemd[];

__device__ __forceinline__ double sinc(double a)
{
    return (a == 0.0) ? 1.0 : sin(a) / a;
}

__global__
void oskar_cudak_correlator_d(const int ns, const int na,
        const double2* k, const double* u, const double* v, const double* l,
        const double* m, const double lambda_bandwidth, double2* vis)
{
}
