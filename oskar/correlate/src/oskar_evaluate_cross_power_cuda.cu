/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include "correlate/oskar_evaluate_cross_power_cuda.h"
#include "correlate/private_correlate_functions_inline.h"

/* Kernels. ================================================================ */

extern __shared__ float2  smem_2f[];
extern __shared__ double2 smem_2d[];
extern __shared__ float4c  smem_4f[];
extern __shared__ double4c smem_4d[];

/* Single precision. */
__global__
void oskar_evaluate_cross_power_cudak_f(const int num_sources,
        const int num_stations, const float4c* restrict jones,
        float4c* restrict out, const float norm)
{
    float4c val1, val2, *p, q;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate cross-power product at the source. */
    p = &smem_4f[threadIdx.x];
    oskar_clear_complex_matrix_f(&val1);
    for (int SP = 0; SP < num_stations; ++SP)
    {
        /* Load data for first station into shared memory. */
        OSKAR_LOAD_MATRIX(smem_4f[threadIdx.x], jones, SP * num_sources + i);
        oskar_clear_complex_matrix_f(&val2);

        /* Cross-correlate. */
        for (int SQ = SP + 1; SQ < num_stations; ++SQ)
        {
            /* Load data for second station into registers. */
            OSKAR_LOAD_MATRIX(q, jones, SQ * num_sources + i);

            /* Multiply-add: val += p * conj(q). */
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.a, p->a, q.a);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.a, p->b, q.b);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.b, p->a, q.c);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.b, p->b, q.d);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.c, p->c, q.a);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.c, p->d, q.b);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.d, p->c, q.c);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2.d, p->d, q.d);
        }

        /* Accumulate partial sum (try to preserve numerical precision). */
        val1.a.x += val2.a.x;
        val1.a.y += val2.a.y;
        val1.b.x += val2.b.x;
        val1.b.y += val2.b.y;
        val1.c.x += val2.c.x;
        val1.c.y += val2.c.y;
        val1.d.x += val2.d.x;
        val1.d.y += val2.d.y;
    }

    /* Calculate average by dividing by number of baselines. */
    val1.a.x *= norm;
    val1.a.y *= norm;
    val1.b.x *= norm;
    val1.b.y *= norm;
    val1.c.x *= norm;
    val1.c.y *= norm;
    val1.d.x *= norm;
    val1.d.y *= norm;

    /* Store result. */
    out[i] = val1;
}

__global__
void oskar_evaluate_cross_power_scalar_cudak_f(
        const int num_sources, const int num_stations,
        const float2* restrict jones, float2* restrict out,
        const float norm)
{
    float2 val1, val2, q;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate cross-power product at the source. */
    val1.x = 0.0f;
    val1.y = 0.0f;
    for (int SP = 0; SP < num_stations; ++SP)
    {
        /* Load data for first station into shared memory. */
        smem_2f[threadIdx.x] = jones[SP * num_sources + i];
        val2.x = 0.0f;
        val2.y = 0.0f;

        /* Cross-correlate. */
        for (int SQ = SP + 1; SQ < num_stations; ++SQ)
        {
            /* Load data for second station into registers. */
            q = jones[SQ * num_sources + i];

            /* Multiply-add: val += p * conj(q). */
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val2, smem_2f[threadIdx.x], q);
        }

        /* Accumulate partial sum (try to preserve numerical precision). */
        val1.x += val2.x;
        val1.y += val2.y;
    }

    /* Calculate average by dividing by number of baselines. */
    val1.x *= norm;
    val1.y *= norm;

    /* Store result. */
    out[i] = val1;
}

/* Double precision. */
__global__
void oskar_evaluate_cross_power_scalar_cudak_d(
        const int num_sources, const int num_stations,
        const double2* restrict jones, double2* restrict out,
        const double norm)
{
    double2 val1, val2, q;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate cross-power product at the source. */
    val1.x = 0.0;
    val1.y = 0.0;
    for (int SP = 0; SP < num_stations; ++SP)
    {
        /* Load data for first station into shared memory. */
        smem_2d[threadIdx.x] = jones[SP * num_sources + i];
        val2.x = 0.0;
        val2.y = 0.0;

        /* Cross-correlate. */
        for (int SQ = SP + 1; SQ < num_stations; ++SQ)
        {
            /* Load data for second station into registers. */
            q = jones[SQ * num_sources + i];

            /* Multiply-add: val += p * conj(q). */
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val1, smem_2d[threadIdx.x], q);
        }

        /* Accumulate partial sum (try to preserve numerical precision). */
        val1.x += val2.x;
        val1.y += val2.y;
    }

    /* Calculate average by dividing by number of baselines. */
    val1.x *= norm;
    val1.y *= norm;

    /* Store result. */
    out[i] = val1;
}

__global__
void oskar_evaluate_cross_power_cudak_d(const int num_sources,
        const int num_stations, const double4c* restrict jones,
        double4c* restrict out, const double norm)
{
    double4c val, *p, q;
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Calculate cross-power product at the source. */
    p = &smem_4d[threadIdx.x];
    oskar_clear_complex_matrix_d(&val);
    for (int SP = 0; SP < num_stations; ++SP)
    {
        /* Load data for first station into shared memory. */
        OSKAR_LOAD_MATRIX(smem_4d[threadIdx.x], jones, SP * num_sources + i);

        /* Cross-correlate. */
        for (int SQ = SP + 1; SQ < num_stations; ++SQ)
        {
            /* Load data for second station into registers. */
            OSKAR_LOAD_MATRIX(q, jones, SQ * num_sources + i);

            /* Multiply-add: val += p * conj(q). */
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.a, p->a, q.a);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.a, p->b, q.b);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.b, p->a, q.c);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.b, p->b, q.d);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.c, p->c, q.a);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.c, p->d, q.b);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.d, p->c, q.c);
            OSKAR_MULTIPLY_ADD_COMPLEX_CONJUGATE(val.d, p->d, q.d);
        }
    }

    /* Calculate average by dividing by number of baselines. */
    val.a.x *= norm;
    val.a.y *= norm;
    val.b.x *= norm;
    val.b.y *= norm;
    val.c.x *= norm;
    val.c.y *= norm;
    val.d.x *= norm;
    val.d.y *= norm;

    /* Store result. */
    out[i] = val;
}

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_cross_power_cuda_f(int num_sources,
        int num_stations, const float4c* d_jones, float4c* d_out)
{
    int num_blocks, num_threads = 128;
    size_t shared_mem = num_threads * sizeof(float4c);
    float norm = 2.0f / (num_stations * (num_stations - 1));
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_cross_power_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (num_sources,
            num_stations, d_jones, d_out, norm);
}

void oskar_evaluate_cross_power_scalar_cuda_f(int num_sources,
        int num_stations, const float2* d_jones, float2* d_out)
{
    int num_blocks, num_threads = 128;
    size_t shared_mem = num_threads * sizeof(float2);
    float norm = 2.0f / (num_stations * (num_stations - 1));
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_cross_power_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (num_sources,
            num_stations, d_jones, d_out, norm);
}

/* Double precision. */
void oskar_evaluate_cross_power_cuda_d(int num_sources,
        int num_stations, const double4c* d_jones, double4c* d_out)
{
    int num_blocks, num_threads = 64;
    size_t shared_mem = num_threads * sizeof(double4c);
    double norm = 2.0 / (num_stations * (num_stations - 1));
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_cross_power_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (num_sources,
            num_stations, d_jones, d_out, norm);
}

void oskar_evaluate_cross_power_scalar_cuda_d(int num_sources,
        int num_stations, const double2* d_jones, double2* d_out)
{
    int num_blocks, num_threads = 128;
    size_t shared_mem = num_threads * sizeof(double2);
    double norm = 2.0 / (num_stations * (num_stations - 1));
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_evaluate_cross_power_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads, shared_mem) (num_sources,
            num_stations, d_jones, d_out, norm);
}

#ifdef __cplusplus
}
#endif
