/*
 * Copyright (c) 2011, The University of Oxford
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

#include "utility/oskar_mem_element_multiply.h"
#include <stdlib.h>

// Single precision.
__global__
static void oskar_cudak_element_multiply_rr_f(int n, const float* a,
        const float* b, float* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

__global__
static void oskar_cudak_element_multiply_cc_f(int n, const float2* a,
        const float2* b, float2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        // Cache the input data.
        float2 ac = a[i];
        float2 bc = b[i];

        // Complex multiply.
        float2 cc;
        cc.x = ac.x * bc.x - ac.y * bc.y; // RE*RE - IM*IM
        cc.y = ac.y * bc.x + ac.x * bc.y; // IM*RE + RE*IM

        c[i] = cc;
    }
}

// Double precision.
__global__
static void oskar_cudak_element_multiply_rr_d(int n, const double* a,
        const double* b, double* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        c[i] = a[i] * b[i];
}

__global__
static void oskar_cudak_element_multiply_cc_d(int n, const double2* a,
        const double2* b, double2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        // Cache the input data.
        double2 ac = a[i];
        double2 bc = b[i];

        // Complex multiply.
        double2 cc;
        cc.x = ac.x * bc.x - ac.y * bc.y; // RE*RE - IM*IM
        cc.y = ac.y * bc.x + ac.x * bc.y; // IM*RE + RE*IM

        c[i] = cc;
    }
}

#ifdef __cplusplus
extern "C"
#endif
int oskar_mem_element_multiply(oskar_Mem* A, const oskar_Mem* B, int num)
{
    if (A == NULL || B == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (A->location != OSKAR_LOCATION_GPU ||
            B->location != OSKAR_LOCATION_GPU)
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    int num_threads = 128; /* FIXME work out what size this should be...? */
    int num_blocks = (num + num_threads - 1) / num_threads;

    if (A->type == OSKAR_DOUBLE &&
            B->type == OSKAR_DOUBLE)
    {
        oskar_cudak_element_multiply_rr_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *A, *B, *A);
    }
    else if (A->type == OSKAR_SINGLE &&
            B->type == OSKAR_SINGLE)
    {
        oskar_cudak_element_multiply_rr_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *A, *B, *A);
    }
    else if (A->type == OSKAR_DOUBLE_COMPLEX &&
            B->type == OSKAR_DOUBLE_COMPLEX)
    {
        oskar_cudak_element_multiply_cc_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *A, *B, *A);
    }
    else if (A->type == OSKAR_SINGLE_COMPLEX &&
            B->type == OSKAR_SINGLE_COMPLEX)
    {
        oskar_cudak_element_multiply_cc_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (num, *A, *B, *A);
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    return OSKAR_SUCCESS;
}
