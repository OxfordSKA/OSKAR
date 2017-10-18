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

#include "mem/oskar_mem_random_uniform_cuda.h"
#include "math/private_random_helpers.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__
void oskar_mem_random_uniform_cudak_f(
        const int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i4 = i * 4;
    if (i4 >= num_elements) return;

    OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

    /* Convert to uniform float. */
    float4 r;
    r.x = oskar_int_to_range_0_to_1_f(u.i[0]);
    r.y = oskar_int_to_range_0_to_1_f(u.i[1]);
    r.z = oskar_int_to_range_0_to_1_f(u.i[2]);
    r.w = oskar_int_to_range_0_to_1_f(u.i[3]);

    /* Store random numbers. */
    if (i4 <= num_elements - 4)
    {
        ((float4*) data)[i] = r;
    }
    else
    {
        /* End case only if length not divisible by 4. */
        data[i4] = r.x;
        if (i4 + 1 < num_elements)
            data[i4 + 1] = r.y;
        if (i4 + 2 < num_elements)
            data[i4 + 2] = r.z;
        if (i4 + 3 < num_elements)
            data[i4 + 3] = r.w;
    }
}

__global__
void oskar_mem_random_uniform_cudak_d(
        const int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i4 = i * 4;
    if (i4 >= num_elements) return;

    OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

    /* Convert to uniform double. */
    double4 r;
    r.x = oskar_int_to_range_0_to_1_d(u.i[0]);
    r.y = oskar_int_to_range_0_to_1_d(u.i[1]);
    r.z = oskar_int_to_range_0_to_1_d(u.i[2]);
    r.w = oskar_int_to_range_0_to_1_d(u.i[3]);

    /* Store random numbers. */
    if (i4 <= num_elements - 4)
    {
        ((double4*) data)[i] = r;
    }
    else
    {
        /* End case only if length not divisible by 4. */
        data[i4] = r.x;
        if (i4 + 1 < num_elements)
            data[i4 + 1] = r.y;
        if (i4 + 2 < num_elements)
            data[i4 + 2] = r.z;
        if (i4 + 3 < num_elements)
            data[i4 + 3] = r.w;
    }
}

void oskar_mem_random_uniform_cuda_f(int num_elements,
        float* d_data, unsigned int seed, unsigned int counter1,
        unsigned int counter2, unsigned int counter3)
{
    int num_blocks, num_threads = 256;
    num_blocks = (((num_elements + 3) / 4) + num_threads - 1) / num_threads;
    oskar_mem_random_uniform_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (num_elements, d_data, seed, counter1, counter2, counter3);
}

void oskar_mem_random_uniform_cuda_d(int num_elements,
        double* d_data, unsigned int seed, unsigned int counter1,
        unsigned int counter2, unsigned int counter3)
{
    int num_blocks, num_threads = 256;
    num_blocks = (((num_elements + 3) / 4) + num_threads - 1) / num_threads;
    oskar_mem_random_uniform_cudak_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (num_elements, d_data, seed, counter1, counter2, counter3);
}

#ifdef __cplusplus
}
#endif
