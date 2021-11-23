/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_enu_directions_to_cirs_relative_directions.h"
#include "convert/oskar_convert_enu_directions_to_cirs_relative_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ENU_DIR_TO_CIRS_REL_DIR(convert_enu_directions_to_cirs_relative_directions_float, float)
OSKAR_CONVERT_ENU_DIR_TO_CIRS_REL_DIR(convert_enu_directions_to_cirs_relative_directions_double, double)

void oskar_convert_enu_directions_to_cirs_relative_directions(
        int offset_in,
        int num_points,
        const oskar_Mem* x,
        const oskar_Mem* y,
        const oskar_Mem* z,
        double ra0_rad,
        double dec0_rad,
        double lon_rad,
        double lat_rad,
        double era_rad,
        double pm_x_rad,
        double pm_y_rad,
        double diurnal_aberration,
        int offset_out,
        oskar_Mem* l,
        oskar_Mem* m,
        oskar_Mem* n,
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
            convert_enu_directions_to_cirs_relative_directions_float(
                    offset_in, num_points,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    sin_ha0_f, cos_ha0_f, sin_dec0_f, cos_dec0_f,
                    sin_lat_f, cos_lat_f,
                    local_pm_x_f, local_pm_y_f, diurnal_aberration_f,
                    offset_out,
                    oskar_mem_float(l, status),
                    oskar_mem_float(m, status),
                    oskar_mem_float(n, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_enu_directions_to_cirs_relative_directions_double(
                    offset_in, num_points,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    sin_ha0, cos_ha0, sin_dec0, cos_dec0,
                    sin_lat, cos_lat,
                    local_pm_x, local_pm_y, diurnal_aberration,
                    offset_out,
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
            k = "convert_enu_directions_to_cirs_relative_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_enu_directions_to_cirs_relative_directions_double";
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
