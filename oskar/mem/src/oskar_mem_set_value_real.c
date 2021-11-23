/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_value_real(oskar_Mem* mem, double value,
        size_t offset, size_t num_elements, int* status)
{
    size_t i = 0;
    if (*status) return;
    const int type = mem->type;
    const int location = mem->location;
    const float value_f = (float) value;
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_DOUBLE:
        {
            double *v = 0;
            v = (double*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i) v[i] = value;
            return;
        }
        case OSKAR_DOUBLE_COMPLEX:
        {
            double2 *v = 0;
            v = (double2*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i)
            {
                v[i].x = value;
                v[i].y = 0.0;
            }
            return;
        }
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
        {
            double4c d;
            double4c *v = 0;
            v = (double4c*)(mem->data) + offset;
            d.a.x = value; d.a.y = 0.0;
            d.b.x = d.b.y = 0.0;
            d.c.x = d.c.y = 0.0;
            d.d.x = value; d.d.y = 0.0;
            for (i = 0; i < num_elements; ++i) v[i] = d;
            return;
        }
        case OSKAR_SINGLE:
        {
            float *v = 0;
            v = (float*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i) v[i] = value_f;
            return;
        }
        case OSKAR_SINGLE_COMPLEX:
        {
            float2 *v = 0;
            v = (float2*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i)
            {
                v[i].x = value_f;
                v[i].y = 0.0f;
            }
            return;
        }
        case OSKAR_SINGLE_COMPLEX_MATRIX:
        {
            float4c d;
            float4c *v = 0;
            v = (float4c*)(mem->data) + offset;
            d.a.x = value_f; d.a.y = 0.0f;
            d.b.x = d.b.y = 0.0f;
            d.c.x = d.c.y = 0.0f;
            d.d.x = value_f; d.d.y = 0.0f;
            for (i = 0; i < num_elements; ++i) v[i] = d;
            return;
        }
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const unsigned int off = (unsigned int) offset;
        const unsigned int n = (unsigned int) num_elements;
        const int is_dbl = (oskar_mem_precision(mem) == OSKAR_DOUBLE);
        const char* k = 0;
        switch (type)
        {
        case OSKAR_DOUBLE:
            k = "mem_set_value_real_r_double"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "mem_set_value_real_c_double"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "mem_set_value_real_m_double"; break;
        case OSKAR_SINGLE:
            k = "mem_set_value_real_r_float"; break;
        case OSKAR_SINGLE_COMPLEX:
            k = "mem_set_value_real_c_float"; break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "mem_set_value_real_m_float"; break;
        default:
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

#ifdef __cplusplus
}
#endif
