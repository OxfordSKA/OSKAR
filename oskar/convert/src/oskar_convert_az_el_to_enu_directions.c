/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_az_el_to_enu_directions.h"
#include "convert/oskar_convert_az_el_to_enu_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_AZ_EL_TO_ENU_DIR(convert_az_el_to_enu_directions_float, float)
OSKAR_CONVERT_AZ_EL_TO_ENU_DIR(convert_az_el_to_enu_directions_double, double)

void oskar_convert_az_el_to_enu_directions(
        int num_points, const oskar_Mem* az_rad, const oskar_Mem* el_rad,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_az_el_to_enu_directions_float(num_points,
                    oskar_mem_float_const(az_rad, status),
                    oskar_mem_float_const(el_rad, status),
                    oskar_mem_float(x, status),
                    oskar_mem_float(y, status),
                    oskar_mem_float(z, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_az_el_to_enu_directions_double(num_points,
                    oskar_mem_double_const(az_rad, status),
                    oskar_mem_double_const(el_rad, status),
                    oskar_mem_double(x, status),
                    oskar_mem_double(y, status),
                    oskar_mem_double(z, status));
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
            k = "convert_az_el_to_enu_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_az_el_to_enu_directions_double";
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
                {PTR_SZ, oskar_mem_buffer_const(az_rad)},
                {PTR_SZ, oskar_mem_buffer_const(el_rad)},
                {PTR_SZ, oskar_mem_buffer(x)},
                {PTR_SZ, oskar_mem_buffer(y)},
                {PTR_SZ, oskar_mem_buffer(z)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
