/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_enu_directions_to_relative_directions.h"
#include "convert/oskar_convert_enu_directions_to_relative_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ENU_DIR_TO_REL_DIR(convert_enu_directions_to_relative_directions_float, float)
OSKAR_CONVERT_ENU_DIR_TO_REL_DIR(convert_enu_directions_to_relative_directions_double, double)

void oskar_convert_enu_directions_to_relative_directions(int offset_in,
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, double ha0, double dec0, double lat,
        int offset_out, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    const double sin_h0  = sin(ha0),  cos_h0  = cos(ha0);
    const double sin_d0  = sin(dec0), cos_d0  = cos(dec0);
    const double sin_lat = sin(lat),  cos_lat = cos(lat);
    const float sin_h0_f = (float) sin_h0, cos_h0_f = (float) cos_h0;
    const float sin_d0_f = (float) sin_d0, cos_d0_f = (float) cos_d0;
    const float sin_lat_f = (float) sin_lat, cos_lat_f = (float) cos_lat;
    if (type != oskar_mem_type(y) || type != oskar_mem_type(z) ||
            type != oskar_mem_type(l) || type != oskar_mem_type(m) ||
            type != oskar_mem_type(n))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(y) ||
            location != oskar_mem_location(z) ||
            location != oskar_mem_location(l) ||
            location != oskar_mem_location(m) ||
            location != oskar_mem_location(n))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_enu_directions_to_relative_directions_float(
                    offset_in, num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    sin_h0_f, cos_h0_f, sin_d0_f, cos_d0_f,
                    sin_lat_f, cos_lat_f, offset_out,
                    oskar_mem_float(l, status),
                    oskar_mem_float(m, status),
                    oskar_mem_float(n, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_enu_directions_to_relative_directions_double(
                    offset_in, num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    sin_h0, cos_h0, sin_d0, cos_d0,
                    sin_lat, cos_lat, offset_out,
                    oskar_mem_double(l, status),
                    oskar_mem_double(m, status),
                    oskar_mem_double(n, status));
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
            k = "convert_enu_directions_to_relative_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_enu_directions_to_relative_directions_double";
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
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_h0 : (const void*)&sin_h0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_h0 : (const void*)&cos_h0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_d0 : (const void*)&sin_d0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_d0 : (const void*)&cos_d0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_lat : (const void*)&sin_lat_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_lat : (const void*)&cos_lat_f},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(l)},
                {PTR_SZ, oskar_mem_buffer(m)},
                {PTR_SZ, oskar_mem_buffer(n)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
