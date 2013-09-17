/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_mem_set_value_real_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_mem_set_value_real_cuda_r_f(int num, float* data, float value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_r_f OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}

void oskar_mem_set_value_real_cuda_c_f(int num, float2* data, float value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_c_f OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}

void oskar_mem_set_value_real_cuda_m_f(int num, float4c* data, float value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_m_f OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}


/* Double precision. */
void oskar_mem_set_value_real_cuda_r_d(int num, double* data, double value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_r_d OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}

void oskar_mem_set_value_real_cuda_c_d(int num, double2* data, double value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_c_d OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}

void oskar_mem_set_value_real_cuda_m_d(int num, double4c* data, double value)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num + num_threads - 1) / num_threads;
    oskar_mem_set_value_real_cudak_m_d OSKAR_CUDAK_CONF(num_blocks,
            num_threads) (num, data, value);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_mem_set_value_real_cudak_r_f(const int num, float* data,
        const float value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i] = value;
}

__global__
void oskar_mem_set_value_real_cudak_c_f(const int num, float2* data,
        const float value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i] = make_float2(value, 0.0f);
}

__global__
void oskar_mem_set_value_real_cudak_m_f(const int num, float4c* data,
        const float value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i].a = make_float2(value, 0.0f);
    data[i].b = make_float2(0.0f, 0.0f);
    data[i].c = make_float2(0.0f, 0.0f);
    data[i].d = make_float2(value, 0.0f);
}


/* Double precision. */
__global__
void oskar_mem_set_value_real_cudak_r_d(const int num, double* data,
        const double value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i] = value;
}

__global__
void oskar_mem_set_value_real_cudak_c_d(const int num, double2* data,
        const double value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i] = make_double2(value, 0.0);
}

__global__
void oskar_mem_set_value_real_cudak_m_d(const int num, double4c* data,
        const double value)
{
    /* Get the array index ID that this thread is working on. */
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num) return;

    /* Set the value. */
    data[i].a = make_double2(value, 0.0);
    data[i].b = make_double2(0.0, 0.0);
    data[i].c = make_double2(0.0, 0.0);
    data[i].d = make_double2(value, 0.0);
}

#ifdef __cplusplus
}
#endif
