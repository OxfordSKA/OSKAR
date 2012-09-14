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

#include "utility/oskar_cuda_check_error.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_element_multiply.h"
#include "utility/oskar_vector_types.h"
#include <stdio.h>
#include <stdlib.h>

/* Single precision. */
__device__
static void oskar_cudaf_complex_multiply_f(const float2& a, const float2& b,
        float2& c)
{
    c.x = a.x * b.x - a.y * b.y; /* RE*RE - IM*IM */
    c.y = a.y * b.x + a.x * b.y; /* IM*RE + RE*IM */
}

__global__
static void oskar_cudak_element_multiply_rr_r_f(const int n, const float* a,
        const float* b, float* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

__global__
static void oskar_cudak_element_multiply_cc_c_f(const int n, const float2* a,
        const float2* b, float2* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        float2 ac, bc, cc;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_f(ac, bc, cc);
        c[i] = cc;
    }
}

__global__
static void oskar_cudak_element_multiply_cc_m_f(const int n, const float2* a,
        const float2* b, float4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        float2 ac, bc, cc;
        float4c m;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_f(ac, bc, cc);

        /* Store result in a matrix. */
        m.a = cc;
        m.b = make_float2(0.0f, 0.0f);
        m.c = make_float2(0.0f, 0.0f);
        m.d = cc;
        c[i] = m;
    }
}

__global__
static void oskar_cudak_element_multiply_cm_m_f(const int n, const float2* a,
        const float4c* b, float4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        float2 ac;
        float4c bc, m;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_f(ac, bc.a, m.a);
        oskar_cudaf_complex_multiply_f(ac, bc.b, m.b);
        oskar_cudaf_complex_multiply_f(ac, bc.c, m.c);
        oskar_cudaf_complex_multiply_f(ac, bc.d, m.d);
        c[i] = m;
    }
}

__global__
static void oskar_cudak_element_multiply_mm_m_f(const int n, const float4c* a,
        const float4c* b, float4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        float4c ac, bc, m;
        float2 t;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_f(ac.a, bc.a, m.a);
        oskar_cudaf_complex_multiply_f(ac.b, bc.c, t);
        m.a.x += t.x;
        m.a.y += t.y;
        oskar_cudaf_complex_multiply_f(ac.a, bc.b, m.b);
        oskar_cudaf_complex_multiply_f(ac.b, bc.d, t);
        m.b.x += t.x;
        m.b.y += t.y;
        oskar_cudaf_complex_multiply_f(ac.c, bc.a, m.c);
        oskar_cudaf_complex_multiply_f(ac.d, bc.c, t);
        m.c.x += t.x;
        m.c.y += t.y;
        oskar_cudaf_complex_multiply_f(ac.c, bc.b, m.d);
        oskar_cudaf_complex_multiply_f(ac.d, bc.d, t);
        m.d.x += t.x;
        m.d.y += t.y;
        c[i] = m;
    }
}

/* Double precision. */
__device__
static void oskar_cudaf_complex_multiply_d(const double2& a, const double2& b,
        double2& c)
{
    c.x = a.x * b.x - a.y * b.y; /* RE*RE - IM*IM */
    c.y = a.y * b.x + a.x * b.y; /* IM*RE + RE*IM */
}

__global__
static void oskar_cudak_element_multiply_rr_r_d(const int n, const double* a,
        const double* b, double* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

__global__
static void oskar_cudak_element_multiply_cc_c_d(const int n, const double2* a,
        const double2* b, double2* c)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        double2 ac, bc, cc;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_d(ac, bc, cc);
        c[i] = cc;
    }
}

__global__
static void oskar_cudak_element_multiply_cc_m_d(const int n, const double2* a,
        const double2* b, double4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        double2 ac, bc, cc;
        double4c m;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_d(ac, bc, cc);

        /* Store result in a matrix. */
        m.a = cc;
        m.b = make_double2(0.0, 0.0);
        m.c = make_double2(0.0, 0.0);
        m.d = cc;
        c[i] = m;
    }
}

__global__
static void oskar_cudak_element_multiply_cm_m_d(const int n, const double2* a,
        const double4c* b, double4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        double2 ac;
        double4c bc, m;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_d(ac, bc.a, m.a);
        oskar_cudaf_complex_multiply_d(ac, bc.b, m.b);
        oskar_cudaf_complex_multiply_d(ac, bc.c, m.c);
        oskar_cudaf_complex_multiply_d(ac, bc.d, m.d);
        c[i] = m;
    }
}

__global__
static void oskar_cudak_element_multiply_mm_m_d(const int n, const double4c* a,
        const double4c* b, double4c* c)
{
    /* Get the array index ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        double4c ac, bc, m;
        double2 t;
        ac = a[i];
        bc = b[i];
        oskar_cudaf_complex_multiply_d(ac.a, bc.a, m.a);
        oskar_cudaf_complex_multiply_d(ac.b, bc.c, t);
        m.a.x += t.x;
        m.a.y += t.y;
        oskar_cudaf_complex_multiply_d(ac.a, bc.b, m.b);
        oskar_cudaf_complex_multiply_d(ac.b, bc.d, t);
        m.b.x += t.x;
        m.b.y += t.y;
        oskar_cudaf_complex_multiply_d(ac.c, bc.a, m.c);
        oskar_cudaf_complex_multiply_d(ac.d, bc.c, t);
        m.c.x += t.x;
        m.c.y += t.y;
        oskar_cudaf_complex_multiply_d(ac.c, bc.b, m.d);
        oskar_cudaf_complex_multiply_d(ac.d, bc.d, t);
        m.d.x += t.x;
        m.d.y += t.y;
        c[i] = m;
    }
}

#ifdef __cplusplus
extern "C"
#endif
void oskar_mem_element_multiply(oskar_Mem* C, oskar_Mem* A, const oskar_Mem* B,
        int num, int* status)
{
    oskar_Mem Ct, At, Bt, *Cp, *Ap;
    const oskar_Mem *Bp;
    int error = OSKAR_ERR_TYPE_MISMATCH; /* Set to type mismatch by default. */

    /* Check all inputs. */
    if (C == NULL) C = A;
    if (!C || !A || !B || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check memory is allocated. */
    if (!A->data || !B->data || !C->data)
    {
        *status = OSKAR_ERR_MEMORY_NOT_ALLOCATED;
        return;
    }

    /* Set the number of elements to multiply. */
    if (num <= 0) num = A->num_elements;

    /* Check that there are enough elements. */
    if (B->num_elements < num || C->num_elements < num)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Copy data to GPU memory if required. */
    oskar_mem_init(&Ct, C->type, OSKAR_LOCATION_GPU, 0, 1, status);
    oskar_mem_init(&At, A->type, OSKAR_LOCATION_GPU, 0, 1, status);
    oskar_mem_init(&Bt, B->type, OSKAR_LOCATION_GPU, 0, 1, status);
    if (C->location != OSKAR_LOCATION_GPU)
    {
        oskar_mem_init(&Ct, C->type, OSKAR_LOCATION_GPU,
                C->num_elements, 1, status);
        Cp = &Ct;
    }
    else
    {
        Cp = C;
    }
    if (A->location != OSKAR_LOCATION_GPU)
    {
        oskar_mem_copy(&At, A, status);
        Ap = &At;
    }
    else
    {
        Ap = A;
    }
    if (B->location != OSKAR_LOCATION_GPU)
    {
        oskar_mem_copy(&Bt, B, status);
        Bp = &Bt;
    }
    else
    {
        Bp = B;
    }

    /* Check if safe to proceed. */
    if (*status) goto cleanup;

    /* Multiply the elements. */
    if (A->type == OSKAR_DOUBLE)
    {
        if (B->type == OSKAR_DOUBLE)
        {
            if (C->type == OSKAR_DOUBLE)
            {
                /* Real, real to real. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_rr_r_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }
    else if (A->type == OSKAR_DOUBLE_COMPLEX)
    {
        if (B->type == OSKAR_DOUBLE_COMPLEX)
        {
            if (C->type == OSKAR_DOUBLE_COMPLEX)
            {
                /* Complex scalar, complex scalar to complex scalar. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cc_c_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
            else if (C->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                /* Complex scalar, complex scalar to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cc_m_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
        else if (B->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            if (C->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                /* Complex scalar, complex matrix to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cm_m_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }
    else if (A->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
    {
        if (B->type == OSKAR_DOUBLE_COMPLEX)
        {
            if (C->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                /* Complex matrix, complex scalar to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cm_m_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Bp, *Ap, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
        else if (B->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            if (C->type == OSKAR_DOUBLE_COMPLEX_MATRIX)
            {
                /* Complex matrix, complex matrix to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_mm_m_d
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }
    else if (A->type == OSKAR_SINGLE)
    {
        if (B->type == OSKAR_SINGLE)
        {
            if (C->type == OSKAR_SINGLE)
            {
                /* Real, real to real. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_rr_r_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }
    else if (A->type == OSKAR_SINGLE_COMPLEX)
    {
        if (B->type == OSKAR_SINGLE_COMPLEX)
        {
            if (C->type == OSKAR_SINGLE_COMPLEX)
            {
                /* Complex scalar, complex scalar to complex scalar. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cc_c_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
            else if (C->type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                /* Complex scalar, complex scalar to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cc_m_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
        else if (B->type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            if (C->type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                /* Complex scalar, complex matrix to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cm_m_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }
    else if (A->type == OSKAR_SINGLE_COMPLEX_MATRIX)
    {
        if (B->type == OSKAR_SINGLE_COMPLEX)
        {
            if (C->type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                /* Complex matrix, complex scalar to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_cm_m_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Bp, *Ap, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
        else if (B->type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            if (C->type == OSKAR_SINGLE_COMPLEX_MATRIX)
            {
                /* Complex matrix, complex matrix to complex matrix. */
                int num_blocks, num_threads = 256;
                num_blocks = (num + num_threads - 1) / num_threads;
                oskar_cudak_element_multiply_mm_m_f
                OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *Ap, *Bp, *Cp);
                oskar_cuda_check_error(status);
                error = 0;
            }
        }
    }

    /* Check for type mismatch error. */
    if (error) *status = error;

    /* Copy result back to host memory if required. */
    if (C->location == OSKAR_LOCATION_CPU)
        oskar_mem_copy(C, Cp, status);

    cleanup:
    oskar_mem_free(&Ct);
    oskar_mem_free(&At);
    oskar_mem_free(&Bt);
}
