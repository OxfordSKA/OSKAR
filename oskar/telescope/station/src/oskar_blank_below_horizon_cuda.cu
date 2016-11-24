/*
 * Copyright (c) 2012, The University of Oxford
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

#include "telescope/station/oskar_blank_below_horizon_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

void oskar_blank_below_horizon_scalar_cuda_f(float2* d_jones,
        int num_sources, const float* d_mask)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_blank_below_horizon_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, d_mask);
}

void oskar_blank_below_horizon_matrix_cuda_f(float4c* d_jones,
        int num_sources, const float* d_mask)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_blank_below_horizon_matrix_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, d_mask);
}

void oskar_blank_below_horizon_scalar_cuda_d(double2* d_jones,
        int num_sources, const double* d_mask)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_blank_below_horizon_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, d_mask);
}

void oskar_blank_below_horizon_matrix_cuda_d(double4c* d_jones,
        int num_sources, const double* d_mask)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_blank_below_horizon_matrix_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, d_mask);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_blank_below_horizon_scalar_cudak_f(float2* jones,
        const int num_sources, const float* mask)
{
    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    if (mask[i] < 0.0f)
        jones[i] = make_float2(0.0f, 0.0f);
}

__global__
void oskar_blank_below_horizon_matrix_cudak_f(float4c* jones,
        const int num_sources, const float* mask)
{
    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    if (mask[i] < 0.0f)
    {
        jones[i].a = make_float2(0.0f, 0.0f);
        jones[i].b = make_float2(0.0f, 0.0f);
        jones[i].c = make_float2(0.0f, 0.0f);
        jones[i].d = make_float2(0.0f, 0.0f);
    }
}

/* Double precision. */
__global__
void oskar_blank_below_horizon_scalar_cudak_d(double2* jones,
        const int num_sources, const double* mask)
{
    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    if (mask[i] < 0.0)
        jones[i] = make_double2(0.0, 0.0);
}

__global__
void oskar_blank_below_horizon_matrix_cudak_d(double4c* jones,
        const int num_sources, const double* mask)
{
    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    if (mask[i] < 0.0)
    {
        jones[i].a = make_double2(0.0, 0.0);
        jones[i].b = make_double2(0.0, 0.0);
        jones[i].c = make_double2(0.0, 0.0);
        jones[i].d = make_double2(0.0, 0.0);
    }
}

#ifdef __cplusplus
}
#endif
