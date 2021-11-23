/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
        int extra_point_at_pole, double delta_phi1, double delta_phi2,
        oskar_Mem* theta, oskar_Mem* phi1, oskar_Mem* phi2, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(theta);
    const int location = oskar_mem_location(theta);
    const float delta_phi1_f = (float) delta_phi1;
    const float delta_phi2_f = (float) delta_phi2;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_enu_directions_to_theta_phi_float(offset_in, num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    extra_point_at_pole, delta_phi1_f, delta_phi2_f,
                    oskar_mem_float(theta, status),
                    oskar_mem_float(phi1, status),
                    oskar_mem_float(phi2, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_enu_directions_to_theta_phi_double(offset_in, num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    extra_point_at_pole, delta_phi1, delta_phi2,
                    oskar_mem_double(theta, status),
                    oskar_mem_double(phi1, status),
                    oskar_mem_double(phi2, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(theta);
        if (type == OSKAR_SINGLE)
        {
            k = "convert_enu_directions_to_theta_phi_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_enu_directions_to_theta_phi_double";
        }
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
                {INT_SZ, &extra_point_at_pole},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&delta_phi1 : (const void*)&delta_phi1_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&delta_phi2 : (const void*)&delta_phi2_f},
                {PTR_SZ, oskar_mem_buffer(theta)},
                {PTR_SZ, oskar_mem_buffer(phi1)},
                {PTR_SZ, oskar_mem_buffer(phi2)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
