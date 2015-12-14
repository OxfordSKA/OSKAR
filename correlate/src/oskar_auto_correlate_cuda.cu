/*
 * Copyright (c) 2015, The University of Oxford
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

#include <private_correlate_functions_inline.h>
#include <oskar_auto_correlate_cuda.h>
#include <oskar_add_inline.h>

/* Kernels. ================================================================ */

extern __shared__ float4c  smem_f[];
extern __shared__ double4c smem_d[];

/* Single precision. */
__global__
void oskar_auto_correlate_cudak_f(const int num_sources,
        const int num_stations, const float4c* restrict jones,
        const float* restrict source_I, const float* restrict source_Q,
        const float* restrict source_U, const float* restrict source_V,
        float4c* restrict vis)
{
    float4c sum;
    int i;

    /* Get station index. */
    const int s = blockDim.y * blockIdx.y + threadIdx.y;

    /* Get pointer to Jones matrix vector for station. */
    const float4c* restrict jones_station = &jones[num_sources * s];

    /* Each thread loops over a subset of the sources. */
    oskar_clear_complex_matrix_f(&sum); /* Partial sum per thread. */
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
        oskar_accumulate_station_visibility_for_source_inline_f(&sum, i,
                source_I, source_Q, source_U, source_V, jones_station);

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_f[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources. */
        for (i = 1; i < blockDim.x; ++i)
            oskar_add_complex_matrix_in_place_f(&sum, &smem_f[i]);

        /* Add result of this thread block to the visibility. */
        /* Blank non-Hermitian values. */
        sum.a.y = 0.0f;
        sum.d.y = 0.0f;
        oskar_add_complex_matrix_in_place_f(&vis[s], &sum);
    }
}

/* Double precision. */
__global__
void oskar_auto_correlate_cudak_d(const int num_sources,
        const int num_stations, const double4c* restrict jones,
        const double* restrict source_I, const double* restrict source_Q,
        const double* restrict source_U, const double* restrict source_V,
        double4c* restrict vis)
{
    double4c sum;
    int i;

    /* Get station index. */
    const int s = blockDim.y * blockIdx.y + threadIdx.y;

    /* Get pointer to Jones matrix vector for station. */
    const double4c* restrict jones_station = &jones[num_sources * s];

    /* Each thread loops over a subset of the sources. */
    oskar_clear_complex_matrix_d(&sum); /* Partial sum per thread. */
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
        oskar_accumulate_station_visibility_for_source_inline_d(&sum, i,
                source_I, source_Q, source_U, source_V, jones_station);

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_d[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources. */
        for (i = 1; i < blockDim.x; ++i)
            oskar_add_complex_matrix_in_place_d(&sum, &smem_d[i]);

        /* Add result of this thread block to the visibility. */
        /* Blank non-Hermitian values. */
        sum.a.y = 0.0;
        sum.d.y = 0.0;
        oskar_add_complex_matrix_in_place_d(&vis[s], &sum);
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_auto_correlate_cuda_f(int num_sources, int num_stations,
        const float4c* d_jones, const float* d_source_I,
        const float* d_source_Q, const float* d_source_U,
        const float* d_source_V, float4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float4c);
    oskar_auto_correlate_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_vis);
}

/* Double precision. */
void oskar_auto_correlate_cuda_d(int num_sources, int num_stations,
        const double4c* d_jones, const double* d_source_I,
        const double* d_source_Q, const double* d_source_U,
        const double* d_source_V, double4c* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double4c);
    oskar_auto_correlate_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_source_Q, d_source_U,
            d_source_V, d_vis);
}

#ifdef __cplusplus
}
#endif
