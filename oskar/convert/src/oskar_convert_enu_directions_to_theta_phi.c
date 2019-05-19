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

#include "convert/define_convert_enu_directions_to_theta_phi.h"
#include "convert/oskar_convert_enu_directions_to_theta_phi.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ENU_DIR_TO_THETA_PHI(convert_enu_directions_to_theta_phi_float, float)
OSKAR_CONVERT_ENU_DIR_TO_THETA_PHI(convert_enu_directions_to_theta_phi_double, double)

void oskar_convert_enu_directions_to_theta_phi(int offset_in, int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double delta_phi, oskar_Mem* theta, oskar_Mem* phi, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(theta);
    const int location = oskar_mem_location(theta);
    const float delta_phi_f = (float) delta_phi;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            convert_enu_directions_to_theta_phi_float(offset_in, num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status), delta_phi_f,
                    oskar_mem_float(theta, status),
                    oskar_mem_float(phi, status));
        else if (type == OSKAR_DOUBLE)
            convert_enu_directions_to_theta_phi_double(offset_in, num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status), delta_phi,
                    oskar_mem_double(theta, status),
                    oskar_mem_double(phi, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(theta);
        if (type == OSKAR_SINGLE)
            k = "convert_enu_directions_to_theta_phi_float";
        else if (type == OSKAR_DOUBLE)
            k = "convert_enu_directions_to_theta_phi_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &offset_in},
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(x)},
                {PTR_SZ, oskar_mem_buffer_const(y)},
                {PTR_SZ, oskar_mem_buffer_const(z)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&delta_phi : (const void*)&delta_phi_f},
                {PTR_SZ, oskar_mem_buffer(theta)},
                {PTR_SZ, oskar_mem_buffer(phi)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
