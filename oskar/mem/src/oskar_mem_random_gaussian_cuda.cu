/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/private_random_helpers.h"
#include "utility/oskar_cuda_registrar.h"
#include <cuda_runtime.h>

__global__
void oskar_mem_random_gaussian_float(
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
OSKAR_CUDA_KERNEL(oskar_mem_random_gaussian_float)

__global__
void oskar_mem_random_gaussian_double(
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
    double r1, r2, r3, r4;
    oskar_box_muller_d(u.i[0], u.i[1], &r1, &r2);
    oskar_box_muller_d(u.i[2], u.i[3], &r3, &r4);
    r1 *= std;
    r2 *= std;
    r3 *= std;
    r4 *= std;

    /* Store random numbers. */
    if (i4 <= num_elements - 4)
    {
        data[i4] = r1;
        data[i4 + 1] = r2;
        data[i4 + 2] = r3;
        data[i4 + 3] = r4;
    }
    else
    {
        /* End case only if length not divisible by 4. */
        data[i4] = r1;
        if (i4 + 1 < num_elements)
        {
            data[i4 + 1] = r2;
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = r3;
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = r4;
        }
    }
}
OSKAR_CUDA_KERNEL(oskar_mem_random_gaussian_double)
