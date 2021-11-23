/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_lon_lat_to_relative_directions.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_LON_LAT_TO_REL_DIR(oskar_convert_lon_lat_to_relative_directions_2d_f, 0, float)
OSKAR_CONVERT_LON_LAT_TO_REL_DIR(oskar_convert_lon_lat_to_relative_directions_2d_d, 0, double)
OSKAR_CONVERT_LON_LAT_TO_REL_DIR(oskar_convert_lon_lat_to_relative_directions_3d_f, 1, float)
OSKAR_CONVERT_LON_LAT_TO_REL_DIR(oskar_convert_lon_lat_to_relative_directions_3d_d, 1, double)

void oskar_convert_lon_lat_to_relative_directions(int num_points,
        const oskar_Mem* lon_rad, const oskar_Mem* lat_rad, double lon0_rad,
        double lat0_rad, oskar_Mem* l, oskar_Mem* m, oskar_Mem* n, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(lon_rad);
    const int location = oskar_mem_location(lon_rad);
    const int is_3d = (n != NULL);
    const double sin_lat0 = sin(lat0_rad);
    const double cos_lat0 = cos(lat0_rad);
    const float lon0_rad_f = (float) lon0_rad;
    const float sin_lat0_f = (float) sin_lat0;
    const float cos_lat0_f = (float) cos_lat0;
    if (oskar_mem_type(lat_rad) != type || oskar_mem_type(l) != type ||
            oskar_mem_type(m) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (oskar_mem_location(lat_rad) != location ||
            oskar_mem_location(l) != location ||
            oskar_mem_location(m) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(lon_rad) < num_points ||
            (int)oskar_mem_length(lat_rad) < num_points)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }
    oskar_mem_ensure(l, num_points, status);
    oskar_mem_ensure(m, num_points, status);
    if (is_3d)
    {
        if (oskar_mem_type(n) != type)
        {
            *status = OSKAR_ERR_TYPE_MISMATCH;
            return;
        }
        if (oskar_mem_location(n) != location)
        {
            *status = OSKAR_ERR_LOCATION_MISMATCH;
            return;
        }
        oskar_mem_ensure(n, num_points, status);
    }
    if (*status) return;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            if (is_3d)
            {
                oskar_convert_lon_lat_to_relative_directions_3d_f(num_points,
                        oskar_mem_float_const(lon_rad, status),
                        oskar_mem_float_const(lat_rad, status),
                        lon0_rad_f, cos_lat0_f, sin_lat0_f,
                        oskar_mem_float(l, status),
                        oskar_mem_float(m, status),
                        oskar_mem_float(n, status));
            }
            else
            {
                oskar_convert_lon_lat_to_relative_directions_2d_f(num_points,
                        oskar_mem_float_const(lon_rad, status),
                        oskar_mem_float_const(lat_rad, status),
                        lon0_rad_f, cos_lat0_f, sin_lat0_f,
                        oskar_mem_float(l, status),
                        oskar_mem_float(m, status), 0);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            if (is_3d)
            {
                oskar_convert_lon_lat_to_relative_directions_3d_d(num_points,
                        oskar_mem_double_const(lon_rad, status),
                        oskar_mem_double_const(lat_rad, status),
                        lon0_rad, cos_lat0, sin_lat0,
                        oskar_mem_double(l, status),
                        oskar_mem_double(m, status),
                        oskar_mem_double(n, status));
            }
            else
            {
                oskar_convert_lon_lat_to_relative_directions_2d_d(num_points,
                        oskar_mem_double_const(lon_rad, status),
                        oskar_mem_double_const(lat_rad, status),
                        lon0_rad, cos_lat0, sin_lat0,
                        oskar_mem_double(l, status),
                        oskar_mem_double(m, status), 0);
            }
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const void* nullp = 0;
        const int is_dbl = oskar_mem_is_double(lon_rad);
        if (type == OSKAR_SINGLE)
        {
            k = is_3d ?
                    "convert_lon_lat_to_relative_directions_3d_float" :
                    "convert_lon_lat_to_relative_directions_2d_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = is_3d ?
                    "convert_lon_lat_to_relative_directions_3d_double" :
                    "convert_lon_lat_to_relative_directions_2d_double";
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
                {PTR_SZ, oskar_mem_buffer_const(lon_rad)},
                {PTR_SZ, oskar_mem_buffer_const(lat_rad)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&lon0_rad : (const void*)&lon0_rad_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_lat0 : (const void*)&cos_lat0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_lat0 : (const void*)&sin_lat0_f},
                {PTR_SZ, oskar_mem_buffer(l)},
                {PTR_SZ, oskar_mem_buffer(m)},
                {PTR_SZ, (is_3d ? oskar_mem_buffer(n) : &nullp)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
