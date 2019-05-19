/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#include "telescope/station/element/define_evaluate_dipole_pattern.h"
#include "telescope/station/element/oskar_evaluate_dipole_pattern.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EVALUATE_DIPOLE_PATTERN(evaluate_dipole_pattern_f, float, float2)
OSKAR_EVALUATE_DIPOLE_PATTERN(evaluate_dipole_pattern_d, double, double2)
OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(evaluate_dipole_pattern_scalar_f, float, float2, float4c)
OSKAR_EVALUATE_DIPOLE_PATTERN_SCALAR(evaluate_dipole_pattern_scalar_d, double, double2, double4c)

void oskar_evaluate_dipole_pattern(int num_points, const oskar_Mem* theta,
        const oskar_Mem* phi, double freq_hz, double dipole_length_m,
        int stride, int offset, oskar_Mem* pattern, int* status)
{
    if (*status) return;
    const int precision = oskar_mem_precision(pattern);
    const int location = oskar_mem_location(pattern);
    const int E_theta_offset = offset;
    const int E_phi_offset = offset + 1;
    const double kL = dipole_length_m * (M_PI * freq_hz / 299792458);
    const double cos_kL = cos(kL);
    const float kL_f = (float) kL;
    const float cos_kL_f = (float) cos_kL;
    if (oskar_mem_location(theta) != location ||
            oskar_mem_location(phi) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(theta) != precision || oskar_mem_type(phi) != precision)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_dipole_pattern_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status),
                    kL_f, cos_kL_f, stride, E_theta_offset, E_phi_offset,
                    oskar_mem_float2(pattern, status),
                    oskar_mem_float2(pattern, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_dipole_pattern_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    kL, cos_kL, stride, E_theta_offset, E_phi_offset,
                    oskar_mem_double2(pattern, status),
                    oskar_mem_double2(pattern, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            evaluate_dipole_pattern_scalar_f(num_points,
                    oskar_mem_float_const(theta, status),
                    oskar_mem_float_const(phi, status),
                    kL_f, cos_kL_f, stride, offset,
                    oskar_mem_float2(pattern, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            evaluate_dipole_pattern_scalar_d(num_points,
                    oskar_mem_double_const(theta, status),
                    oskar_mem_double_const(phi, status),
                    kL, cos_kL, stride, offset,
                    oskar_mem_double2(pattern, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(pattern);
        switch (oskar_mem_type(pattern))
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_dipole_pattern_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_dipole_pattern_double"; break;
        case OSKAR_SINGLE_COMPLEX:
            k = "evaluate_dipole_pattern_scalar_float"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "evaluate_dipole_pattern_scalar_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        if (oskar_mem_is_matrix(pattern))
        {
            const oskar_Arg args[] = {
                    {INT_SZ, &num_points},
                    {PTR_SZ, oskar_mem_buffer_const(theta)},
                    {PTR_SZ, oskar_mem_buffer_const(phi)},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&kL : (const void*)&kL_f},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&cos_kL : (const void*)&cos_kL_f},
                    {INT_SZ, &stride},
                    {INT_SZ, &E_theta_offset},
                    {INT_SZ, &E_phi_offset},
                    {PTR_SZ, oskar_mem_buffer(pattern)},
                    {PTR_SZ, oskar_mem_buffer(pattern)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
        else
        {
            const oskar_Arg args[] = {
                    {INT_SZ, &num_points},
                    {PTR_SZ, oskar_mem_buffer_const(theta)},
                    {PTR_SZ, oskar_mem_buffer_const(phi)},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&kL : (const void*)&kL_f},
                    {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                            (const void*)&cos_kL : (const void*)&cos_kL_f},
                    {INT_SZ, &stride},
                    {INT_SZ, &offset},
                    {PTR_SZ, oskar_mem_buffer(pattern)}
            };
            oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                    sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
