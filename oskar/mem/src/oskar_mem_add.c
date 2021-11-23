/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "utility/oskar_device.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_add(
        oskar_Mem* out,
        const oskar_Mem* in1,
        const oskar_Mem* in2,
        size_t offset_out,
        size_t offset_in1,
        size_t offset_in2,
        size_t num_elements,
        int* status)
{
    oskar_Mem *a_temp = 0, *b_temp = 0;
    const oskar_Mem *a_ = 0, *b_ = 0; /* Pointers. */
    if (num_elements == 0) return;
    const int type = oskar_mem_type(out);
    const int precision = oskar_mem_precision(out);
    const int location = oskar_mem_location(out);
    if (*status) return;
    if (type != oskar_mem_type(in1) || type != oskar_mem_type(in2))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    a_ = in1;
    b_ = in2;
    if (oskar_mem_location(in1) != location)
    {
        a_temp = oskar_mem_create_copy(in1, location, status);
        a_ = a_temp;
    }
    if (oskar_mem_location(in2) != location)
    {
        b_temp = oskar_mem_create_copy(in2, location, status);
        b_ = b_temp;
    }
    if (oskar_mem_is_matrix(out))  { offset_out *= 4; num_elements *= 4; }
    if (oskar_mem_is_complex(out)) { offset_out *= 2; num_elements *= 2; }
    if (oskar_mem_is_matrix(in1))    offset_in1 *= 4;
    if (oskar_mem_is_complex(in1))   offset_in1 *= 2;
    if (oskar_mem_is_matrix(in2))    offset_in2 *= 4;
    if (oskar_mem_is_complex(in2))   offset_in2 *= 2;
    if (location == OSKAR_CPU)
    {
        size_t i = 0;
        if (precision == OSKAR_DOUBLE)
        {
            double *c = oskar_mem_double(out, status);
            const double *a = oskar_mem_double_const(a_, status);
            const double *b = oskar_mem_double_const(b_, status);
            for (i = 0; i < num_elements; ++i)
            {
                c[i + offset_out] = a[i + offset_in1] + b[i + offset_in2];
            }
        }
        else if (precision == OSKAR_SINGLE)
        {
            float *c = oskar_mem_float(out, status);
            const float *a = oskar_mem_float_const(a_, status);
            const float *b = oskar_mem_float_const(b_, status);
            for (i = 0; i < num_elements; ++i)
            {
                c[i + offset_out] = a[i + offset_in1] + b[i + offset_in2];
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const unsigned int off_a = (unsigned int) offset_in1;
        const unsigned int off_b = (unsigned int) offset_in2;
        const unsigned int off_c = (unsigned int) offset_out;
        const unsigned int n = (unsigned int) num_elements;
        const char* k = 0;
        if (precision == OSKAR_DOUBLE)
        {
            k = "mem_add_double";
        }
        else if (precision == OSKAR_SINGLE)
        {
            k = "mem_add_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off_a},
                {INT_SZ, &off_b},
                {INT_SZ, &off_c},
                {INT_SZ, &n},
                {PTR_SZ, oskar_mem_buffer_const(a_)},
                {PTR_SZ, oskar_mem_buffer_const(b_)},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }

    /* Free temporary arrays. */
    oskar_mem_free(a_temp, status);
    oskar_mem_free(b_temp, status);
}

#ifdef __cplusplus
}
#endif
