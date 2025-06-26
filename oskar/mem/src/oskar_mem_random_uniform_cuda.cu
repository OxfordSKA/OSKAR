/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/private_random_helpers.h"
#include "utility/oskar_cuda_registrar.h"
#include <cuda_runtime.h>

__global__
void oskar_mem_random_uniform_float(
        const unsigned int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i4 = i * 4;
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
        {
            data[i4 + 1] = r.y;
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = r.z;
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = r.w;
        }
    }
}
OSKAR_CUDA_KERNEL(oskar_mem_random_uniform_float)

__global__
void oskar_mem_random_uniform_double(
        const unsigned int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int i4 = i * 4;
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
        {
            data[i4 + 1] = r.y;
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = r.z;
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = r.w;
        }
    }
}
OSKAR_CUDA_KERNEL(oskar_mem_random_uniform_double)
