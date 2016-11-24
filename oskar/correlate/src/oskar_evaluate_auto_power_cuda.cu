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

#include "correlate/oskar_evaluate_auto_power_cuda.h"
#include "correlate/private_correlate_functions_inline.h"

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_auto_power_cudak_f(const int num_sources,
        const float4c* restrict jones, float4c* restrict out)
{
    float4c val1, val2;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate auto-power product at the source. */
    OSKAR_LOAD_MATRIX(val1, jones, i);
    val2 = val1;
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_f(&val1, &val2);

    /* Store result. */
    out[i] = val1;
}

__global__
void oskar_evaluate_auto_power_scalar_cudak_f(const int num_sources,
        const float2* restrict jones, float2* restrict out)
{
    float2 val1, val2;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate auto-power product at the source. */
    val1 = jones[i];
    val2 = val1;
    oskar_multiply_complex_conjugate_in_place_f(&val1, &val2);

    /* Store result. */
    out[i] = val1;
}

/* Double precision. */
__global__
void oskar_evaluate_auto_power_cudak_d(const int num_sources,
        const double4c* restrict jones, double4c* restrict out)
{
    double4c val1, val2;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate auto-power product at the source. */
    OSKAR_LOAD_MATRIX(val1, jones, i);
    val2 = val1;
    oskar_multiply_complex_matrix_conjugate_transpose_in_place_d(&val1, &val2);

    /* Store result. */
    out[i] = val1;
}

__global__
void oskar_evaluate_auto_power_scalar_cudak_d(const int num_sources,
        const double2* restrict jones, double2* restrict out)
{
    double2 val1, val2;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate auto-power product at the source. */
    val1 = jones[i];
    val2 = val1;
    oskar_multiply_complex_conjugate_in_place_d(&val1, &val2);

    /* Store result. */
    out[i] = val1;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_auto_power_cuda_f(int num_sources,
        const float4c* d_jones, float4c* d_out)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_auto_power_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, d_jones, d_out);
}

void oskar_evaluate_auto_power_scalar_cuda_f(int num_sources,
        const float2* d_jones, float2* d_out)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_auto_power_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, d_jones, d_out);
}

/* Double precision. */
void oskar_evaluate_auto_power_cuda_d(int num_sources,
        const double4c* d_jones, double4c* d_out)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_auto_power_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, d_jones, d_out);
}

void oskar_evaluate_auto_power_scalar_cuda_d(int num_sources,
        const double2* d_jones, double2* d_out)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_auto_power_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_sources, d_jones, d_out);
}

#ifdef __cplusplus
}
#endif
