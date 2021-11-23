/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "math/private_random_helpers.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

static void mem_random_uniform_float(
        const unsigned int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    unsigned int i = 0, i4 = 0, n1 = 0;
    n1 = num_elements / 4;
    for (i = 0; i < n1; ++i)
    {
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = i * 4;
        data[i4]     = oskar_int_to_range_0_to_1_f(u.i[0]);
        data[i4 + 1] = oskar_int_to_range_0_to_1_f(u.i[1]);
        data[i4 + 2] = oskar_int_to_range_0_to_1_f(u.i[2]);
        data[i4 + 3] = oskar_int_to_range_0_to_1_f(u.i[3]);
    }
    if (num_elements % 4)
    {
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = n1 * 4;
        data[i4] = oskar_int_to_range_0_to_1_f(u.i[0]);
        if (i4 + 1 < num_elements)
        {
            data[i4 + 1] = oskar_int_to_range_0_to_1_f(u.i[1]);
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = oskar_int_to_range_0_to_1_f(u.i[2]);
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = oskar_int_to_range_0_to_1_f(u.i[3]);
        }
    }
}

static void mem_random_uniform_double(
        const unsigned int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3)
{
    unsigned int i = 0, i4 = 0, n1 = 0;
    n1 = num_elements / 4;
    for (i = 0; i < n1; ++i)
    {
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = i * 4;
        data[i4]     = oskar_int_to_range_0_to_1_d(u.i[0]);
        data[i4 + 1] = oskar_int_to_range_0_to_1_d(u.i[1]);
        data[i4 + 2] = oskar_int_to_range_0_to_1_d(u.i[2]);
        data[i4 + 3] = oskar_int_to_range_0_to_1_d(u.i[3]);
    }
    if (num_elements % 4)
    {
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to uniform float. */
        i4 = n1 * 4;
        data[i4] = oskar_int_to_range_0_to_1_d(u.i[0]);
        if (i4 + 1 < num_elements)
        {
            data[i4 + 1] = oskar_int_to_range_0_to_1_d(u.i[1]);
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = oskar_int_to_range_0_to_1_d(u.i[2]);
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = oskar_int_to_range_0_to_1_d(u.i[3]);
        }
    }
}

void oskar_mem_random_uniform(oskar_Mem* data, unsigned int seed,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        int* status)
{
    unsigned int num_elements = 0;
    if (*status) return;
    const int type = oskar_mem_precision(data);
    const int location = oskar_mem_location(data);
    num_elements = (unsigned int)oskar_mem_length(data);
    if (oskar_mem_is_complex(data)) num_elements *= 2;
    if (oskar_mem_is_matrix(data)) num_elements *= 4;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            mem_random_uniform_float(num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3);
        }
        else if (type == OSKAR_DOUBLE)
        {
            mem_random_uniform_double(num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3);
        }
    }
    else
    {
        const char* k = 0;
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        if (type == OSKAR_SINGLE)
        {
            k = "mem_random_uniform_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "mem_random_uniform_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = ((((num_elements + 3) / 4) + local_size[0] - 1) /
                local_size[0]) * local_size[0];
        const oskar_Arg args[] = {
                {INT_SZ, &num_elements},
                {PTR_SZ, oskar_mem_buffer(data)},
                {INT_SZ, &seed},
                {INT_SZ, &counter1},
                {INT_SZ, &counter2},
                {INT_SZ, &counter3}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
