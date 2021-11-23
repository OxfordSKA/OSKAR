/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_relative_directions_to_enu_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_REL_DIR_TO_ENU_DIR(convert_relative_directions_to_enu_directions_float, float)
OSKAR_CONVERT_REL_DIR_TO_ENU_DIR(convert_relative_directions_to_enu_directions_double, double)

void oskar_convert_relative_directions_to_enu_directions(int at_origin,
        int bypass, int offset_in, int num_points, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, double ha0, double dec0,
        double lat, int offset_out, oskar_Mem* x, oskar_Mem* y, oskar_Mem* z,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    const double sin_h0  = sin(ha0),  cos_h0  = cos(ha0);
    const double sin_d0  = sin(dec0), cos_d0  = cos(dec0);
    const double sin_lat = sin(lat),  cos_lat = cos(lat);
    const float sin_h0_f = (float) sin_h0, cos_h0_f = (float) cos_h0;
    const float sin_d0_f = (float) sin_d0, cos_d0_f = (float) cos_d0;
    const float sin_lat_f  = (float) sin_lat, cos_lat_f  = (float) cos_lat;
    if (type != oskar_mem_type(y) || type != oskar_mem_type(z))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (location != oskar_mem_location(y) ||
            location != oskar_mem_location(z))
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_relative_directions_to_enu_directions_float(
                    at_origin, bypass, offset_in, num_points,
                    (!at_origin ? oskar_mem_float_const(l, status) : 0),
                    (!at_origin ? oskar_mem_float_const(m, status) : 0),
                    (!at_origin ? oskar_mem_float_const(n, status) : 0),
                    sin_h0_f, cos_h0_f, sin_d0_f, cos_d0_f,
                    sin_lat_f, cos_lat_f, offset_out,
                    oskar_mem_float(x, status),
                    oskar_mem_float(y, status),
                    oskar_mem_float(z, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_relative_directions_to_enu_directions_double(
                    at_origin, bypass, offset_in, num_points,
                    (!at_origin ? oskar_mem_double_const(l, status) : 0),
                    (!at_origin ? oskar_mem_double_const(m, status) : 0),
                    (!at_origin ? oskar_mem_double_const(n, status) : 0),
                    sin_h0, cos_h0, sin_d0, cos_d0,
                    sin_lat, cos_lat, offset_out,
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
        const int is_dbl = (type == OSKAR_DOUBLE);
        const char* k = 0;
        const void* np = 0;
        if (type == OSKAR_SINGLE)
        {
            k = "convert_relative_directions_to_enu_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_relative_directions_to_enu_directions_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        if (num_points <= 32)
        {
            local_size[0] = 32;
        }
        else if (num_points <= 64)
        {
            local_size[0] = 64;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &at_origin},
                {INT_SZ, &bypass},
                {INT_SZ, &offset_in},
                {INT_SZ, &num_points},
                {PTR_SZ, (!at_origin ? oskar_mem_buffer_const(l) : &np)},
                {PTR_SZ, (!at_origin ? oskar_mem_buffer_const(m) : &np)},
                {PTR_SZ, (!at_origin ? oskar_mem_buffer_const(n) : &np)},
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
