/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_enu_directions_to_az_el.h"
#include "convert/oskar_convert_enu_directions_to_az_el.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ENU_DIR_TO_AZ_EL(convert_enu_directions_to_az_el_float, float)
OSKAR_CONVERT_ENU_DIR_TO_AZ_EL(convert_enu_directions_to_az_el_double, double)

void oskar_convert_enu_directions_to_az_el(
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_Mem* az_rad, oskar_Mem* el_rad, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_enu_directions_to_az_el_float(num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    oskar_mem_float(az_rad, status),
                    oskar_mem_float(el_rad, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_enu_directions_to_az_el_double(num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    oskar_mem_double(az_rad, status),
                    oskar_mem_double(el_rad, status));
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
        if (type == OSKAR_SINGLE)
        {
            k = "convert_enu_directions_to_az_el_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_enu_directions_to_az_el_double";
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
                {PTR_SZ, oskar_mem_buffer(az_rad)},
                {PTR_SZ, oskar_mem_buffer(el_rad)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
