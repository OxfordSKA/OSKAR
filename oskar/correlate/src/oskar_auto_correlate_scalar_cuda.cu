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

#include "correlate/oskar_auto_correlate_scalar_cuda.h"
#include <cuda_runtime.h>

template <typename REAL, typename REAL2>
__global__
void oskar_acorr_scalar_cudak(
        const int                   num_sources,
        const int                   num_stations,
        const REAL2* const restrict jones,
        const REAL*  const restrict source_I,
        REAL2*             restrict vis)
{
    extern __shared__ __align__(sizeof(double)) unsigned char my_smem[];
    REAL* smem = reinterpret_cast<REAL*>(my_smem); // Allows template.
    const int s = blockIdx.y; // Station index.
    const REAL2* const restrict jones_station = &jones[num_sources * s];
    REAL sum = (REAL) 0;
    for (int i = threadIdx.x; i < num_sources; i += blockDim.x)
    {
        const REAL2 t = jones_station[i];
        sum += (t.x * t.x + t.y * t.y) * source_I[i];
    }
    smem[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x == 0)
    {
        for (int i = 1; i < blockDim.x; ++i) sum += smem[i];
        vis[s].x += sum;
    }
}

void oskar_auto_correlate_scalar_cuda_f(int num_sources, int num_stations,
        const float2* d_jones, const float* d_source_I, float2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(float);
    oskar_acorr_scalar_cudak<float, float2>
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_vis);
}

void oskar_auto_correlate_scalar_cuda_d(int num_sources, int num_stations,
        const double2* d_jones, const double* d_source_I, double2* d_vis)
{
    dim3 num_threads(128, 1);
    dim3 num_blocks(1, num_stations);
    size_t shared_mem = num_threads.x * sizeof(double);
    oskar_acorr_scalar_cudak<double, double2>
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem)
    (num_sources, num_stations, d_jones, d_source_I, d_vis);
}
