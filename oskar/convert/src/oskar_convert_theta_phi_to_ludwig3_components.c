/*
 * Copyright (c) 2020, The University of Oxford
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

#include "convert/define_convert_theta_phi_to_ludwig3_components.h"
#include "convert/oskar_convert_theta_phi_to_ludwig3_components.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3(convert_theta_phi_to_ludwig3_float, float, float2, float4c)
OSKAR_CONVERT_THETA_PHI_TO_LUDWIG3(convert_theta_phi_to_ludwig3_double, double, double2, double4c)

void oskar_convert_theta_phi_to_ludwig3_components(
        int num_points, const oskar_Mem* phi_x, const oskar_Mem* phi_y,
        int offset, oskar_Mem* jones, int* status)
{
    if (*status) return;
    const int type = oskar_mem_precision(jones);
    const int location = oskar_mem_location(jones);
    if (!oskar_mem_is_matrix(jones))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            convert_theta_phi_to_ludwig3_float(num_points,
                    oskar_mem_float_const(phi_x, status),
                    oskar_mem_float_const(phi_y, status), offset,
                    oskar_mem_float4c(jones, status));
        else if (type == OSKAR_DOUBLE)
            convert_theta_phi_to_ludwig3_double(num_points,
                    oskar_mem_double_const(phi_x, status),
                    oskar_mem_double_const(phi_y, status), offset,
                    oskar_mem_double4c(jones, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE)
            k = "convert_theta_phi_to_ludwig3_float";
        else if (type == OSKAR_DOUBLE)
            k = "convert_theta_phi_to_ludwig3_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(phi_x)},
                {PTR_SZ, oskar_mem_buffer_const(phi_y)},
                {INT_SZ, &offset},
                {PTR_SZ, oskar_mem_buffer(jones)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
