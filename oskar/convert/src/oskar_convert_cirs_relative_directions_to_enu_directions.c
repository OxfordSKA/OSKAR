/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_cirs_relative_directions_to_enu_directions.h"
#include "convert/oskar_convert_cirs_relative_directions_to_enu_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_CIRS_REL_DIR_TO_ENU_DIR(convert_cirs_relative_directions_to_enu_directions_float, float)
OSKAR_CONVERT_CIRS_REL_DIR_TO_ENU_DIR(convert_cirs_relative_directions_to_enu_directions_double, double)

void oskar_convert_cirs_relative_directions_to_enu_directions(
        int at_origin,
        int bypass,
        int offset_in,
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        const oskar_Mem* n,
        double ra0_rad,
        double dec0_rad,
        double lon_rad,
        double lat_rad,
        double era_rad,
        double pm_x_rad,
        double pm_y_rad,
        double diurnal_aberration,
        int offset_out,
        oskar_Mem* x,
        oskar_Mem* y,
        oskar_Mem* z,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    const double ha0_rad = era_rad + lon_rad - ra0_rad;
    const double sin_ha0  = sin(ha0_rad);
    const double cos_ha0  = cos(ha0_rad);
    const double sin_dec0 = sin(dec0_rad);
    const double cos_dec0 = cos(dec0_rad);
    const double sin_lon  = sin(lon_rad);
    const double cos_lon  = cos(lon_rad);
    const double sin_lat  = sin(lat_rad);
    const double cos_lat  = cos(lat_rad);
    const double local_pm_x = pm_x_rad * cos_lon - pm_y_rad * sin_lon;
    const double local_pm_y = pm_x_rad * sin_lon + pm_y_rad * cos_lon;
    const float sin_ha0_f  = (float) sin_ha0;
    const float cos_ha0_f  = (float) cos_ha0;
    const float sin_dec0_f = (float) sin_dec0;
    const float cos_dec0_f = (float) cos_dec0;
    const float sin_lat_f  = (float) sin_lat;
    const float cos_lat_f  = (float) cos_lat;
    const float local_pm_x_f = (float) local_pm_x;
    const float local_pm_y_f = (float) local_pm_y;
    const float diurnal_aberration_f = (float) diurnal_aberration;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
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
            convert_cirs_relative_directions_to_enu_directions_float(
                    at_origin, bypass, offset_in, num_points,
                    (!at_origin ? oskar_mem_float_const(l, status) : 0),
                    (!at_origin ? oskar_mem_float_const(m, status) : 0),
                    (!at_origin ? oskar_mem_float_const(n, status) : 0),
                    sin_ha0_f, cos_ha0_f, sin_dec0_f, cos_dec0_f,
                    sin_lat_f, cos_lat_f, local_pm_x_f, local_pm_y_f,
                    diurnal_aberration_f, offset_out,
                    oskar_mem_float(x, status),
                    oskar_mem_float(y, status),
                    oskar_mem_float(z, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_cirs_relative_directions_to_enu_directions_double(
                    at_origin, bypass, offset_in, num_points,
                    (!at_origin ? oskar_mem_double_const(l, status) : 0),
                    (!at_origin ? oskar_mem_double_const(m, status) : 0),
                    (!at_origin ? oskar_mem_double_const(n, status) : 0),
                    sin_ha0, cos_ha0, sin_dec0, cos_dec0,
                    sin_lat, cos_lat, local_pm_x, local_pm_y,
                    diurnal_aberration, offset_out,
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
            k = "convert_cirs_relative_directions_to_enu_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_cirs_relative_directions_to_enu_directions_double";
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
                        (const void*)&sin_ha0 : (const void*)&sin_ha0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_ha0 : (const void*)&cos_ha0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_dec0 : (const void*)&sin_dec0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_dec0 : (const void*)&cos_dec0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_lat : (const void*)&sin_lat_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_lat : (const void*)&cos_lat_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&local_pm_x : (const void*)&local_pm_x_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&local_pm_y : (const void*)&local_pm_y_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&diurnal_aberration :
                        (const void*)&diurnal_aberration_f},
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
