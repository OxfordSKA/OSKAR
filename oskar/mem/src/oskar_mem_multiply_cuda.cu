/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#include "math/oskar_multiply_inline.h"
#include "mem/oskar_mem_multiply_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
__global__
void oskar_mem_multiply_cudak_rr_r_f(const int n, const float* a,
        const float* b, float* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] * b[i];
}

__global__
void oskar_mem_multiply_cudak_cc_c_f(const int n, const float2* a,
        const float2* b, float2* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float2 ac, bc, cc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_f(&cc, &ac, &bc);
    c[i] = cc;
}

__global__
void oskar_mem_multiply_cudak_cc_m_f(const int n, const float2* a,
        const float2* b, float4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float2 ac, bc, cc;
    float4c m;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_f(&cc, &ac, &bc);

    /* Store result in a matrix. */
    m.a = cc;
    m.b.x = 0.0f;
    m.b.y = 0.0f;
    m.c.x = 0.0f;
    m.c.y = 0.0f;
    m.d = cc;
    c[i] = m;
}

__global__
void oskar_mem_multiply_cudak_cm_m_f(const int n, const float2* a,
        const float4c* b, float4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float2 ac;
    float4c bc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_matrix_complex_scalar_in_place_f(&bc, &ac);
    c[i] = bc;
}

__global__
void oskar_mem_multiply_cudak_mm_m_f(const int n, const float4c* a,
        const float4c* b, float4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    float4c ac, bc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_matrix_in_place_f(&ac, &bc);
    c[i] = ac;
}

void oskar_mem_multiply_cuda_rr_r_f(int num, float* d_c,
        const float* d_a, const float* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_rr_r_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cc_c_f(int num, float2* d_c,
        const float2* d_a, const float2* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cc_c_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cc_m_f(int num, float4c* d_c,
        const float2* d_a, const float2* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cc_m_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cm_m_f(int num, float4c* d_c,
        const float2* d_a, const float4c* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cm_m_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_mm_m_f(int num, float4c* d_c,
        const float4c* d_a, const float4c* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_mm_m_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}


/* Double precision. */
__global__
void oskar_mem_multiply_cudak_rr_r_d(const int n, const double* a,
        const double* b, double* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    c[i] = a[i] * b[i];
}

__global__
void oskar_mem_multiply_cudak_cc_c_d(const int n, const double2* a,
        const double2* b, double2* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double2 ac, bc, cc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_d(&cc, &ac, &bc);
    c[i] = cc;
}

__global__
void oskar_mem_multiply_cudak_cc_m_d(const int n, const double2* a,
        const double2* b, double4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double2 ac, bc, cc;
    double4c m;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_d(&cc, &ac, &bc);

    /* Store result in a matrix. */
    m.a = cc;
    m.b.x = 0.0;
    m.b.y = 0.0;
    m.c.x = 0.0;
    m.c.y = 0.0;
    m.d = cc;
    c[i] = m;
}

__global__
void oskar_mem_multiply_cudak_cm_m_d(const int n, const double2* a,
        const double4c* b, double4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double2 ac;
    double4c bc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_matrix_complex_scalar_in_place_d(&bc, &ac);
    c[i] = bc;
}

__global__
void oskar_mem_multiply_cudak_mm_m_d(const int n, const double4c* a,
        const double4c* b, double4c* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double4c ac, bc;
    ac = a[i];
    bc = b[i];
    oskar_multiply_complex_matrix_in_place_d(&ac, &bc);
    c[i] = ac;
}

void oskar_mem_multiply_cuda_rr_r_d(int num, double* d_c,
        const double* d_a, const double* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_rr_r_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cc_c_d(int num, double2* d_c,
        const double2* d_a, const double2* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cc_c_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cc_m_d(int num, double4c* d_c,
        const double2* d_a, const double2* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cc_m_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_cm_m_d(int num, double4c* d_c,
        const double2* d_a, const double4c* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_cm_m_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

void oskar_mem_multiply_cuda_mm_m_d(int num, double4c* d_c,
        const double4c* d_a, const double4c* d_b)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_multiply_cudak_mm_m_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, d_a, d_b, d_c);
}

#ifdef __cplusplus
}
#endif
