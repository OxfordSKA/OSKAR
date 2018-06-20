/*
 * Copyright (c) 2015-2018, The University of Oxford
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

#include "correlate/private_correlate_functions_inline.h"
#include "correlate/oskar_auto_correlate_cuda.h"
#include "math/oskar_add_inline.h"
#include <cuda_runtime.h>

template <typename REAL, typename REAL2, typename REAL8>
__global__
void oskar_acorr_cudak(
        const int                   num_sources,
        const int                   num_stations,
        const REAL8* const restrict jones,
        const REAL*  const restrict source_I,
        const REAL*  const restrict source_Q,
        const REAL*  const restrict source_U,
        const REAL*  const restrict source_V,
        REAL8*             restrict vis)
{
    extern __shared__ __align__(sizeof(double4c)) unsigned char my_smem[];
    REAL8* smem = reinterpret_cast<REAL8*>(my_smem); // Allows template.
    const int s = blockIdx.y; // Station index.
    const REAL8* const restrict jones_station = &jones[num_sources * s];
    REAL8 m1, m2, sum;
    OSKAR_CLEAR_COMPLEX_MATRIX(REAL, sum)
    for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        // Construct source brightness matrix.
        OSKAR_CONSTRUCT_B(REAL, m2,
                source_I[i], source_Q[i], source_U[i], source_V[i])

        // Multiply first Jones matrix with source brightness matrix.
        OSKAR_LOAD_MATRIX(m1, jones_station[i])
        OSKAR_MUL_COMPLEX_MATRIX_HERMITIAN_IN_PLACE(REAL2, m1, m2);

        // Multiply result with second (Hermitian transposed) Jones matrix.
        OSKAR_LOAD_MATRIX(m2, jones_station[i])
        OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(REAL2, m1, m2);

        // Accumulate.
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, m1)
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < blockDim.x; ++i)
            OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(sum, smem[i])

        // Blank non-Hermitian values.
        sum.a.y = (REAL)0; sum.d.y = (REAL)0;
        OSKAR_ADD_COMPLEX_MATRIX_IN_PLACE(vis[s], sum);
    }
}

void oskar_auto_correlate_cuda_f(int num_sources, int num_stations,
        const float4c* d_jones, const float* d_source_I,
        const float* d_source_Q, const float* d_source_U,
        const float* d_source_V, float4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float4c);
    oskar_acorr_cudak<float, float2, float4c>
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_vis);
}

void oskar_auto_correlate_cuda_d(int num_sources, int num_stations,
        const double4c* d_jones, const double* d_source_I,
        const double* d_source_Q, const double* d_source_U,
        const double* d_source_V, double4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double4c);
    oskar_acorr_cudak<double, double2, double4c>
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_vis);
}
