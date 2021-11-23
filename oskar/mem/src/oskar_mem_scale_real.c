/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C"
#endif
void oskar_mem_scale_real(oskar_Mem* mem, double value,
        size_t offset, size_t num_elements, int* status)
{
    if (*status) return;
    const int location = mem->location;
    const int precision = oskar_type_precision(mem->type);
    if (oskar_type_is_complex(mem->type))
    {
        offset *= 2;
        num_elements *= 2;
    }
    if (oskar_type_is_matrix(mem->type))
    {
        offset *= 4;
        num_elements *= 4;
    }
    if (location == OSKAR_CPU)
    {
        size_t i = 0;
        if (precision == OSKAR_SINGLE)
        {
            float *aa = ((float*) mem->data) + offset;
            const float value_f = (float) value;
            for (i = 0; i < num_elements; ++i) aa[i] *= value_f;
        }
        else if (precision == OSKAR_DOUBLE)
        {
            double *aa = ((double*) mem->data) + offset;
            for (i = 0; i < num_elements; ++i) aa[i] *= value;
        }
        else *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const unsigned int n = (unsigned int) num_elements;
        const unsigned int off = (unsigned int) offset;
        const float value_f = (float) value;
        const int is_dbl = (precision == OSKAR_DOUBLE);
        const char* k = 0;
        if (precision == OSKAR_SINGLE)
        {
            k = "mem_scale_float";
        }
        else if (precision == OSKAR_DOUBLE)
        {
            k = "mem_scale_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(num_elements, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &off},
                {INT_SZ, &n},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&value : (const void*)&value_f},
                {PTR_SZ, oskar_mem_buffer(mem)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}
