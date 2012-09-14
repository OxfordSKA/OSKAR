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

#include <cuda_runtime_api.h>
#include "utility/oskar_mem_set_value_real.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_vector_types.h"

/* Single precision. */
__global__
static void oskar_cudak_set_value_real_r_f(const int n, float* p,
        const float scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i] = scalar;
}

__global__
static void oskar_cudak_set_value_real_c_f(const int n, float2* p,
        const float scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i] = make_float2(scalar, 0.0f);
}

__global__
static void oskar_cudak_set_value_real_m_f(const int n, float4c* p,
        const float scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i].a = make_float2(scalar, 0.0f);
    p[i].b = make_float2(0.0f, 0.0f);
    p[i].c = make_float2(0.0f, 0.0f);
    p[i].d = make_float2(scalar, 0.0f);
}


/* Double precision. */
__global__
static void oskar_cudak_set_value_real_r_d(const int n, double* p,
        const double scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i] = scalar;
}

__global__
static void oskar_cudak_set_value_real_c_d(const int n, double2* p,
        const double scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i] = make_double2(scalar, 0.0);
}

__global__
static void oskar_cudak_set_value_real_m_d(const int n, double4c* p,
        const double scalar)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Set the value. */
    p[i].a = make_double2(scalar, 0.0);
    p[i].b = make_double2(0.0, 0.0);
    p[i].c = make_double2(0.0, 0.0);
    p[i].d = make_double2(scalar, 0.0);
}

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_value_real(oskar_Mem* mem, double val, int* status)
{
    int i, n, type, location;

    /* Check all inputs. */
    if (!mem || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the data type, location, and number of elements. */
    type = mem->type;
    location = mem->location;
    n = mem->num_elements;

    if (location == OSKAR_LOCATION_CPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            double *v;
            v = (double*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = val;
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            double2 *v;
            v = (double2*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = make_double2(val, 0.0);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            double4c d;
            double4c *v;
            v = (double4c*)(mem->data);
            for (i = 0; i < n; ++i)
            {
                d.a = make_double2(val, 0.0);
                d.b = make_double2(0.0, 0.0);
                d.c = make_double2(0.0, 0.0);
                d.d = make_double2(val, 0.0);
                v[i] = d;
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *v;
            v = (float*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = val;
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            float2 *v;
            v = (float2*)(mem->data);
            for (i = 0; i < n; ++i) v[i] = make_float2(val, 0.0f);
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            float4c d;
            float4c *v;
            v = (float4c*)(mem->data);
            for (i = 0; i < n; ++i)
            {
                d.a = make_float2(val, 0.0f);
                d.b = make_float2(0.0f, 0.0f);
                d.c = make_float2(0.0f, 0.0f);
                d.d = make_float2(val, 0.0f);
                v[i] = d;
            }
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else if (location == OSKAR_LOCATION_GPU)
    {
        if (type == OSKAR_DOUBLE)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_r_d OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (double*)(mem->data), val);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_c_d OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (double2*)(mem->data), val);
        }
        else if (type == OSKAR_DOUBLE_COMPLEX_MATRIX)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_m_d OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (double4c*)(mem->data), val);
        }
        else if (type == OSKAR_SINGLE)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_r_f OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (float*)(mem->data), (float)val);
        }
        else if (type == OSKAR_SINGLE_COMPLEX)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_c_f OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (float2*)(mem->data), (float)val);
        }
        else if (type == OSKAR_SINGLE_COMPLEX_MATRIX)
        {
            int num_blocks, num_threads = 256;
            num_blocks = (n + num_threads - 1) / num_threads;
            oskar_cudak_set_value_real_m_f OSKAR_CUDAK_CONF(num_blocks,
                    num_threads) (n, (float4c*)(mem->data), (float)val);
        }
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
        *status = OSKAR_ERR_BAD_LOCATION;
}

#ifdef __cplusplus
}
#endif
