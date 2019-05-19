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

#include "convert/define_convert_ludwig3_to_theta_phi_components.h"
#include "convert/oskar_convert_ludwig3_to_theta_phi_components.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI(convert_ludwig3_to_theta_phi_float, float, float2)
OSKAR_CONVERT_LUDWIG3_TO_THETA_PHI(convert_ludwig3_to_theta_phi_double, double, double2)

void oskar_convert_ludwig3_to_theta_phi_components(
        int num_points, const oskar_Mem* phi, int stride, int offset,
        oskar_Mem* vec, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(phi);
    const int location = oskar_mem_location(phi);
    const int off_h = offset, off_v = offset + 1;
    if (!oskar_mem_is_matrix(vec))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            convert_ludwig3_to_theta_phi_float(
                    num_points,
                    oskar_mem_float_const(phi, status),
                    stride, off_h, off_v,
                    oskar_mem_float2(vec, status),
                    oskar_mem_float2(vec, status));
        else if (type == OSKAR_DOUBLE)
            convert_ludwig3_to_theta_phi_double(
                    num_points,
                    oskar_mem_double_const(phi, status),
                    stride, off_h, off_v,
                    oskar_mem_double2(vec, status),
                    oskar_mem_double2(vec, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE)
            k = "convert_ludwig3_to_theta_phi_float";
        else if (type == OSKAR_DOUBLE)
            k = "convert_ludwig3_to_theta_phi_double";
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
                {PTR_SZ, oskar_mem_buffer_const(phi)},
                {INT_SZ, &stride},
                {INT_SZ, &off_h},
                {INT_SZ, &off_v},
                {PTR_SZ, oskar_mem_buffer(vec)},
                {PTR_SZ, oskar_mem_buffer(vec)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
