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

#include "interferometry/cudak/oskar_cudak_correlator_scalar.h"
#include "math/cudak/oskar_cudaf_mul_c_c_conj.h"
#include "math/cudak/oskar_cudaf_sinc.h"


#define ONE_OVER_2C 1.66782047599076024788E-9   // 1 / (2c)
#define ONE_OVER_2Cf 1.66782047599076024788E-9f // 1 / (2c)

// Indices into the visibility/baseline matrix.
#define AI blockIdx.x // Column index.
#define AJ blockIdx.y // Row index.

extern __shared__ float2  smem_f[];
extern __shared__ double2 smem_d[];

// Single precision.
__global__
void oskar_cudak_correlator_scalar_f(const int ns, const int na,
        const float2* jones, const float* b, const float* u,
        const float* v, const float* l,
        const float* m, const float lambda_bandwidth, float2* vis)
{
    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (AJ >= AI) return;

    // Common things per thread block.
    __device__ __shared__ float uu, vv;
    if (threadIdx.x == 0)
    {
        // Determine UV-distance for baseline (common per thread block).
        uu = ONE_OVER_2Cf * lambda_bandwidth * (u[AI] - u[AJ]);
        vv = ONE_OVER_2Cf * lambda_bandwidth * (v[AI] - v[AJ]);
    }
    __syncthreads();

    // Get pointers to both source vectors for station i and j.
    const float2* station_i = &jones[ns * AI];
    const float2* station_j = &jones[ns * AJ];

    // Each thread loops over a subset of the sources.
    {
        float2 sum = make_float2(0.0f, 0.0f); // Partial sum per thread.
        for (int t = threadIdx.x; t < ns; t += blockDim.x)
        {
            // Compute bandwidth-smearing term first (register optimisation).
            float rb = oskar_cudaf_sinc_f(uu * l[t] + vv * m[t]);
            float2 c_a = station_i[t];
            float2 c_b = station_j[t];
            float2 temp;

            // Complex-conjugate multiply.
            oskar_cudaf_mul_c_c_conj_f(c_a, c_b, temp);

            // Multiply by the source brightness.
            temp.x *= b[t];
            temp.y *= b[t];

            // Multiply result by bandwidth-smearing term.
            sum.x += temp.x * rb;
            sum.y += temp.y * rb;
        }
        smem_f[threadIdx.x] = sum;
    }
    __syncthreads();

    // Accumulate contents of shared memory.
    if (threadIdx.x == 0)
    {
        // Sum over all sources for this baseline.
        float2 sum = make_float2(0.0f, 0.0f);
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.x += smem_f[i].x;
            sum.y += smem_f[i].y;
        }

        // Determine 1D index.
        int idx = AJ*(na-1) - (AJ-1)*AJ/2 + AI - AJ - 1;

        // Modify existing visibility.
        vis[idx].x += sum.x;
        vis[idx].y += sum.y;
    }
}

// Double precision.
__global__
void oskar_cudak_correlator_scalar_d(const int ns, const int na,
        const double2* jones, const double* b, const double* u,
        const double* v, const double* l,
        const double* m, const double lambda_bandwidth, double2* vis)
{
    // Return immediately if we're in the lower triangular half of the
    // visibility matrix.
    if (AJ >= AI) return;

    // Common things per thread block.
    __device__ __shared__ double uu, vv;
    if (threadIdx.x == 0)
    {
        // Determine UV-distance for baseline (common per thread block).
        uu = ONE_OVER_2C * lambda_bandwidth * (u[AI] - u[AJ]);
        vv = ONE_OVER_2C * lambda_bandwidth * (v[AI] - v[AJ]);
    }
    __syncthreads();

    // Get pointers to both source vectors for station i and j.
    const double2* station_i = &jones[ns * AI];
    const double2* station_j = &jones[ns * AJ];

    // Each thread loops over a subset of the sources.
    {
        double2 sum = make_double2(0.0, 0.0); // Partial sum per thread.
        for (int t = threadIdx.x; t < ns; t += blockDim.x)
        {
            // Compute bandwidth-smearing term first (register optimisation).
            double rb = oskar_cudaf_sinc_d(uu * l[t] + vv * m[t]);
            double2 c_a = station_i[t];
            double2 c_b = station_j[t];
            double2 temp;

            // Complex-conjugate multiply.
            oskar_cudaf_mul_c_c_conj_d(c_a, c_b, temp);

            // Multiply by the source brightness.
            temp.x *= b[t];
            temp.y *= b[t];

            // Multiply result by bandwidth-smearing term.
            sum.x += temp.x * rb;
            sum.y += temp.y * rb;
        }
        smem_d[threadIdx.x] = sum;
    }
    __syncthreads();

    // Accumulate contents of shared memory.
    if (threadIdx.x == 0)
    {
        // Sum over all sources for this baseline.
        double2 sum = make_double2(0.0, 0.0);
        for (int i = 0; i < blockDim.x; ++i)
        {
            sum.x += smem_d[i].x;
            sum.y += smem_d[i].y;
        }

        // Determine 1D index.
        int idx = AJ*(na-1) - (AJ-1)*AJ/2 + AI - AJ - 1;

        // Modify existing visibility.
        vis[idx].x += sum.x;
        vis[idx].y += sum.y;
    }
}
