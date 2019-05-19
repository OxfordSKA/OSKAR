/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include "math/define_gaussian_circular.h"
#include "math/oskar_gaussian_circular.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_GAUSSIAN_CIRCULAR_COMPLEX(gaussian_circular_complex_f, float, float2)
OSKAR_GAUSSIAN_CIRCULAR_MATRIX(gaussian_circular_matrix_f, float, float4c)
OSKAR_GAUSSIAN_CIRCULAR_COMPLEX(gaussian_circular_complex_d, double, double2)
OSKAR_GAUSSIAN_CIRCULAR_MATRIX(gaussian_circular_matrix_d, double, double4c)

void oskar_gaussian_circular(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, double std,
        oskar_Mem* out, int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(out);
    const double inv_2_var = 1.0 / (2.0 * std * std);
    const float inv_2_var_f = (float)inv_2_var;
    if (oskar_mem_precision(out) != oskar_mem_type(l) ||
            oskar_mem_precision(out) != oskar_mem_type(m))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(l) || location != oskar_mem_location(m))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(l) < num_points ||
            (int)oskar_mem_length(m) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    oskar_mem_ensure(out, num_points, status);
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(out))
        {
        case OSKAR_SINGLE_COMPLEX:
            gaussian_circular_complex_f(num_points,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status), inv_2_var_f,
                    oskar_mem_float2(out, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            gaussian_circular_complex_d(num_points,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status), inv_2_var,
                    oskar_mem_double2(out, status));
            break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            gaussian_circular_matrix_f(num_points,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status), inv_2_var_f,
                    oskar_mem_float4c(out, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            gaussian_circular_matrix_d(num_points,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status), inv_2_var,
                    oskar_mem_double4c(out, status));
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
        const int is_dbl = oskar_mem_is_double(out);
        switch (oskar_mem_type(out))
        {
        case OSKAR_SINGLE_COMPLEX:
            k = "gaussian_circular_complex_float"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "gaussian_circular_complex_double"; break;
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "gaussian_circular_matrix_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "gaussian_circular_matrix_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&inv_2_var : (const void*)&inv_2_var_f},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
