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

extern __shared__ float2  smem_f[];
extern __shared__ double2 smem_d[];

/* Single precision. */
__global__
void oskar_auto_correlate_scalar_cudak_f(const int num_sources,
        const int num_stations, const float2* restrict jones,
        const float* restrict source_I, float2* restrict vis)
{
    float2 sum;
    int i;

    /* Get station index. */
    const int s = blockDim.y * blockIdx.y + threadIdx.y;

    /* Get pointer to Jones matrix vector for station. */
    const float2* restrict jones_station = &jones[num_sources * s];

    /* Each thread loops over a subset of the sources. */
    sum.x = 0.0f;
    sum.y = 0.0f;
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
        oskar_accumulate_station_visibility_for_source_scalar_inline_f(
                &sum, i, source_I, jones_station);

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_f[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources. We only need the real part. */
        for (i = 1; i < blockDim.x; ++i)
            sum.x += smem_f[i].x;

        /* Add result of this thread block to the visibility. */
        vis[s].x += sum.x;
    }
}

/* Double precision. */
__global__
void oskar_auto_correlate_scalar_cudak_d(const int num_sources,
        const int num_stations, const double2* restrict jones,
        const double* restrict source_I, double2* restrict vis)
{
    double2 sum;
    int i;

    /* Get station index. */
    const int s = blockDim.y * blockIdx.y + threadIdx.y;

    /* Get pointer to Jones matrix vector for station. */
    const double2* restrict jones_station = &jones[num_sources * s];

    /* Each thread loops over a subset of the sources. */
    sum.x = 0.0;
    sum.y = 0.0;
    for (i = threadIdx.x; i < num_sources; i += blockDim.x)
        oskar_accumulate_station_visibility_for_source_scalar_inline_d(
                &sum, i, source_I, jones_station);

    /* Store partial sum for the thread in shared memory and synchronise. */
    smem_d[threadIdx.x] = sum;
    __syncthreads();

    /* Accumulate contents of shared memory. */
    if (threadIdx.x == 0)
    {
        /* Sum over all sources. We only need the real part. */
        for (i = 1; i < blockDim.x; ++i)
            sum.x += smem_d[i].x;

        /* Add result of this thread block to the visibility. */
        vis[s].x += sum.x;
    }
}

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_auto_correlate_scalar_cuda_f(int num_sources, int num_stations,
        const float2* d_jones, const float* d_source_I, float2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float2);
    oskar_auto_correlate_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_vis);
}

/* Double precision. */
void oskar_auto_correlate_scalar_cuda_d(int num_sources, int num_stations,
        const double2* d_jones, const double* d_source_I, double2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double2);
    oskar_auto_correlate_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_vis);
}

#ifdef __cplusplus
}
#endif
