/*
 * Copyright (c) 2015-2019, The University of Oxford
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

#include "math/private_random_helpers.h"
#include "utility/oskar_cuda_registrar.h"
#include <cuda_runtime.h>

__global__
void mem_random_gaussian_float(
        const unsigned int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const float std)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i4 = i * 4;
    if (i4 >= num_elements) return;

    OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

    /* Convert to normalised Gaussian distribution. */
    float4 r;
    oskar_box_muller_f(u.i[0], u.i[1], &r.x, &r.y);
    oskar_box_muller_f(u.i[2], u.i[3], &r.z, &r.w);
    r.x = std * r.x;
    r.y = std * r.y;
    r.z = std * r.z;
    r.w = std * r.w;

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
OSKAR_CUDA_KERNEL(mem_random_gaussian_float)

__global__
void mem_random_gaussian_double(
        const unsigned int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const double std)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i4 = i * 4;
    if (i4 >= num_elements) return;

    OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

    /* Convert to normalised Gaussian distribution. */
    double4 r;
    oskar_box_muller_d(u.i[0], u.i[1], &r.x, &r.y);
    oskar_box_muller_d(u.i[2], u.i[3], &r.z, &r.w);
    r.x = std * r.x;
    r.y = std * r.y;
    r.z = std * r.z;
    r.w = std * r.w;

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
OSKAR_CUDA_KERNEL(mem_random_gaussian_double)
