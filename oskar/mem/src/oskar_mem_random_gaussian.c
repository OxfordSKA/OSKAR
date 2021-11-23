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

static void mem_random_gaussian_float(
        const unsigned int num_elements, float* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const float std)
{
    unsigned int i = 0, i4 = 0, n1 = 0;
    n1 = num_elements / 4;
    for (i = 0; i < n1; ++i)
    {
        float t[4];
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = i * 4;
        oskar_box_muller_f(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_f(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]     = t[0];
        data[i4 + 1] = t[1];
        data[i4 + 2] = t[2];
        data[i4 + 3] = t[3];
    }
    if (num_elements % 4)
    {
        float t[4];
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = n1 * 4;
        oskar_box_muller_f(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_f(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4] = t[0];
        if (i4 + 1 < num_elements)
        {
            data[i4 + 1] = t[1];
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = t[2];
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = t[3];
        }
    }
}

static void mem_random_gaussian_double(
        const unsigned int num_elements, double* data,
        const unsigned int seed, const unsigned int counter1,
        const unsigned int counter2, const unsigned int counter3,
        const double std)
{
    unsigned int i = 0, i4 = 0, n1 = 0;
    n1 = num_elements / 4;
    for (i = 0; i < n1; ++i)
    {
        double t[4];
        OSKAR_R123_GENERATE_4(seed, i, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = i * 4;
        oskar_box_muller_d(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_d(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4]     = t[0];
        data[i4 + 1] = t[1];
        data[i4 + 2] = t[2];
        data[i4 + 3] = t[3];
    }
    if (num_elements % 4)
    {
        double t[4];
        OSKAR_R123_GENERATE_4(seed, n1, counter1, counter2, counter3)

        /* Convert to normalised Gaussian distribution. */
        i4 = n1 * 4;
        oskar_box_muller_d(u.i[0], u.i[1], &t[0], &t[1]);
        oskar_box_muller_d(u.i[2], u.i[3], &t[2], &t[3]);
        t[0] = std * t[0];
        t[1] = std * t[1];
        t[2] = std * t[2];
        t[3] = std * t[3];

        /* Store. */
        data[i4] = t[0];
        if (i4 + 1 < num_elements)
        {
            data[i4 + 1] = t[1];
        }
        if (i4 + 2 < num_elements)
        {
            data[i4 + 2] = t[2];
        }
        if (i4 + 3 < num_elements)
        {
            data[i4 + 3] = t[3];
        }
    }
}

void oskar_mem_random_gaussian(oskar_Mem* data, unsigned int seed,
        unsigned int counter1, unsigned int counter2, unsigned int counter3,
        double std, int* status)
{
    unsigned int num_elements = 0;
    if (*status) return;
    const float std_f = (float) std;
    const int type = oskar_mem_precision(data);
    const int location = oskar_mem_location(data);
    num_elements = (unsigned int)oskar_mem_length(data);
    if (oskar_mem_is_complex(data)) num_elements *= 2;
    if (oskar_mem_is_matrix(data)) num_elements *= 4;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            mem_random_gaussian_float(num_elements,
                    oskar_mem_float(data, status), seed,
                    counter1, counter2, counter3, std_f);
        }
        else if (type == OSKAR_DOUBLE)
        {
            mem_random_gaussian_double(num_elements,
                    oskar_mem_double(data, status), seed,
                    counter1, counter2, counter3, std);
        }
    }
    else
    {
        const char* k = 0;
        const int is_dbl = (type == OSKAR_DOUBLE);
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        if (type == OSKAR_SINGLE)
        {
            k = "mem_random_gaussian_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "mem_random_gaussian_double";
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
                {INT_SZ, &counter3},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&std : (const void*)&std_f}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
