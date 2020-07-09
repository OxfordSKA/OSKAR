/*
 * Copyright (c) 2014-2020, The University of Oxford
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

#include "correlate/define_correlate_utils.h"
#include "correlate/define_evaluate_cross_power.h"
#include "correlate/oskar_evaluate_cross_power.h"
#include "math/define_multiply.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CROSS_POWER_MATRIX(evaluate_cross_power_float, float, float2, float4c)
OSKAR_CROSS_POWER_MATRIX(evaluate_cross_power_double, double, double2, double4c)
OSKAR_CROSS_POWER_SCALAR(evaluate_cross_power_scalar_float, float, float2)
OSKAR_CROSS_POWER_SCALAR(evaluate_cross_power_scalar_double, double, double2)

void oskar_evaluate_cross_power(int num_sources, int num_stations,
        const oskar_Mem* jones,
        double src_I, double src_Q, double src_U, double src_V,
        int offset_out, oskar_Mem* out, int *status)
{
    if (*status) return;
    const int type = oskar_mem_type(jones);
    const int location = oskar_mem_location(jones);
    const double norm = 2.0 / (num_stations * (num_stations - 1));
    const float norm_f = (float) norm;
    const float src_I_f = (float) src_I;
    const float src_Q_f = (float) src_Q;
    const float src_U_f = (float) src_U;
    const float src_V_f = (float) src_V;
    if (type != oskar_mem_type(out))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(out))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            evaluate_cross_power_float(num_sources, num_stations,
                    oskar_mem_float4c_const(jones, status),
                    src_I_f, src_Q_f, src_U_f, src_V_f,
                    norm, offset_out,
                    oskar_mem_float4c(out, status));
            break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            evaluate_cross_power_double(num_sources, num_stations,
                    oskar_mem_double4c_const(jones, status),
                    src_I, src_Q, src_U, src_V,
                    norm, offset_out,
                    oskar_mem_double4c(out, status));
            break;
        case OSKAR_SINGLE_COMPLEX:
            evaluate_cross_power_scalar_float(num_sources, num_stations,
                    oskar_mem_float2_const(jones, status),
                    src_I_f, src_Q_f, src_U_f, src_V_f,
                    norm, offset_out,
                    oskar_mem_float2(out, status));
            break;
        case OSKAR_DOUBLE_COMPLEX:
            evaluate_cross_power_scalar_double(num_sources, num_stations,
                    oskar_mem_double2_const(jones, status),
                    src_I, src_Q, src_U, src_V,
                    norm, offset_out,
                    oskar_mem_double2(out, status));
            break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {128, 1, 1}, global_size[] = {1, 1, 1};
        const int is_dbl = (oskar_type_precision(type) == OSKAR_DOUBLE);
        const char* k = 0;
        switch (type)
        {
        case OSKAR_SINGLE_COMPLEX_MATRIX:
            k = "evaluate_cross_power_float"; break;
        case OSKAR_DOUBLE_COMPLEX_MATRIX:
            k = "evaluate_cross_power_double";
            local_size[0] = 64;
            break;
        case OSKAR_SINGLE_COMPLEX:
            k = "evaluate_cross_power_scalar_float"; break;
        case OSKAR_DOUBLE_COMPLEX:
            k = "evaluate_cross_power_scalar_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {INT_SZ, &num_stations},
                {PTR_SZ, oskar_mem_buffer_const(jones)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&src_I : (const void*)&src_I_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&src_Q : (const void*)&src_Q_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&src_U : (const void*)&src_U_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&src_V : (const void*)&src_V_f},
                {is_dbl ? DBL_SZ : FLT_SZ,
                        is_dbl ? (const void*)&norm : (const void*)&norm_f},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
