/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#include "telescope/station/element/oskar_apply_element_taper_cosine.h"
#include "telescope/station/element/define_apply_element_taper_cosine.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_ELEMENT_TAPER_COSINE_SCALAR(element_taper_cosine_scalar_f, float, float2)
OSKAR_ELEMENT_TAPER_COSINE_MATRIX(element_taper_cosine_matrix_f, float, float4c)
OSKAR_ELEMENT_TAPER_COSINE_SCALAR(element_taper_cosine_scalar_d, double, double2)
OSKAR_ELEMENT_TAPER_COSINE_MATRIX(element_taper_cosine_matrix_d, double, double4c)

void oskar_apply_element_taper_cosine(int num_sources, double cos_power,
        const oskar_Mem* theta, int offset_out, oskar_Mem* jones, int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(jones);
    const float cos_power_f = (float) cos_power;
    if (oskar_mem_location(theta) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(theta) != oskar_mem_precision(jones))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(jones))
        {
        case OSKAR_SINGLE_COMPLEX:
            element_taper_cosine_scalar_f(num_sources, cos_power_f,
                    oskar_mem_float_const(theta, status), offset_out,
                    oskar_mem_float2(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            element_taper_cosine_scalar_d(num_sources, cos_power,
                    oskar_mem_double_const(theta, status), offset_out,
                    oskar_mem_double2(jones, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            element_taper_cosine_matrix_f(num_sources, cos_power_f,
                    oskar_mem_float_const(theta, status), offset_out,
                    oskar_mem_float4c(jones, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            element_taper_cosine_matrix_d(num_sources, cos_power,
                    oskar_mem_double_const(theta, status), offset_out,
                    oskar_mem_double4c(jones, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            break;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(jones);
        switch (oskar_mem_type(jones))
        {
        case OSKAR_SINGLE_COMPLEX:
            k = "apply_element_taper_cosine_scalar_float"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "apply_element_taper_cosine_scalar_double"; break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "apply_element_taper_cosine_matrix_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "apply_element_taper_cosine_matrix_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_power : (const void*)&cos_power_f},
                {PTR_SZ, oskar_mem_buffer_const(theta)},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(jones)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
