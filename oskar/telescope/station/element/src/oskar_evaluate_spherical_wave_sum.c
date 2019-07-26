/*
 * Copyright (c) 2019, The University of Oxford
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

#include "telescope/station/element/oskar_evaluate_spherical_wave_sum.h"
#include "telescope/station/element/define_evaluate_spherical_wave.h"
#include "log/oskar_log.h"
#include "math/define_legendre_polynomial.h"
#include "math/define_multiply.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EVALUATE_SPHERICAL_WAVE_SUM(evaluate_spherical_wave_sum_float, float, float2, float4c)
OSKAR_EVALUATE_SPHERICAL_WAVE_SUM(evaluate_spherical_wave_sum_double, double, double2, double4c)

void oskar_evaluate_spherical_wave_sum(int num_points, const oskar_Mem* theta,
        const oskar_Mem* phi_x, const oskar_Mem* phi_y, int l_max,
        const oskar_Mem* alpha, int offset, oskar_Mem* pattern, int* status)
{
    if (*status) return;
    const int location = oskar_mem_location(pattern);
    const int coeff_required = (l_max + 1) * (l_max + 1) - 1;
    if (oskar_mem_length(alpha) < (size_t) coeff_required)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_float(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi_x, status),
                    oskar_mem_float_const(phi_y, status), l_max,
                    oskar_mem_float4c_const(alpha, status), offset,
                    oskar_mem_float4c(pattern, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_spherical_wave_sum_double(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi_x, status),
                    oskar_mem_double_const(phi_y, status), l_max,
                    oskar_mem_double4c_const(alpha, status), offset,
                    oskar_mem_double4c(pattern, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
        case OSKAR_DOUBLE_COMPLEX:
            oskar_log_error(0, "Spherical wave patterns cannot be used "
                    "in scalar mode");
            *status = OSKAR_ERR_BAD_DATA_TYPE;
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
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_spherical_wave_sum_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_spherical_wave_sum_double"; break;
        case OSKAR_SINGLE_COMPLEX:
        case OSKAR_DOUBLE_COMPLEX:
            oskar_log_error(0, "Spherical wave patterns cannot be used "
                    "in scalar mode");
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg arg[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(theta)},
                {PTR_SZ, oskar_mem_buffer_const(phi_x)},
                {PTR_SZ, oskar_mem_buffer_const(phi_y)},
                {INT_SZ, &l_max},
                {PTR_SZ, oskar_mem_buffer_const(alpha)},
                {INT_SZ, &offset},
                {PTR_SZ, oskar_mem_buffer(pattern)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(arg) / sizeof(oskar_Arg), arg, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
