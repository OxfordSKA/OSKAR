/*
 * Copyright (c) 2012-2020, The University of Oxford
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

#include "mem/oskar_mem.h"
#include "mem/private_mem.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_mem_set_value_real(oskar_Mem* mem, double value,
        size_t offset, size_t num_elements, int* status)
{
    size_t i;
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
            double *v;
            v = (double*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i) v[i] = value;
            return;
        }
        case OSKAR_DOUBLE_COMPLEX:
        {
            double2 *v;
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
            double4c *v;
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
            float *v;
            v = (float*)(mem->data) + offset;
            for (i = 0; i < num_elements; ++i) v[i] = value_f;
            return;
        }
        case OSKAR_SINGLE_COMPLEX:
        {
            float2 *v;
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
            float4c *v;
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
