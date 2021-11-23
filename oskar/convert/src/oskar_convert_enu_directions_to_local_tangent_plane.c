/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_enu_directions_to_local_tangent_plane.h"
#include "convert/oskar_convert_enu_directions_to_local_tangent_plane.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ENU_DIR_TO_LOCAL(convert_enu_directions_to_local_float, float)
OSKAR_CONVERT_ENU_DIR_TO_LOCAL(convert_enu_directions_to_local_double, double)

void oskar_convert_enu_directions_to_local_tangent_plane(int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ref_az_rad, double ref_el_rad, oskar_Mem* l, oskar_Mem* m,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    const double phi = ref_az_rad - M_PI;
    const double theta = ref_el_rad - (M_PI / 2.0);
    const double cos_phi = cos(phi);
    const double sin_phi = sin(phi);
    const double cos_theta = cos(theta);
    const double sin_theta = sin(theta);
    const float cos_phi_f = (float) cos_phi;
    const float sin_phi_f = (float) sin_phi;
    const float cos_theta_f = (float) cos_theta;
    const float sin_theta_f = (float) sin_theta;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_enu_directions_to_local_float(num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    cos_phi_f, sin_phi_f, cos_theta_f, sin_theta_f,
                    oskar_mem_float(l, status),
                    oskar_mem_float(m, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_enu_directions_to_local_double(num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    cos_phi, sin_phi, cos_theta, sin_theta,
                    oskar_mem_double(l, status),
                    oskar_mem_double(m, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const int is_dbl = (type == OSKAR_DOUBLE);
        const char* k = 0;
        if (type == OSKAR_SINGLE)
        {
            k = "convert_enu_directions_to_local_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_enu_directions_to_local_double";
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
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(x)},
                {PTR_SZ, oskar_mem_buffer_const(y)},
                {PTR_SZ, oskar_mem_buffer_const(z)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_phi : (const void*)&cos_phi_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_phi : (const void*)&sin_phi_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_theta : (const void*)&cos_theta_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_theta : (const void*)&sin_theta_f},
                {PTR_SZ, oskar_mem_buffer(l)},
                {PTR_SZ, oskar_mem_buffer(m)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
