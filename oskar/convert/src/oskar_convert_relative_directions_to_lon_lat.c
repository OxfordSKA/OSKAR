/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_relative_directions_to_lon_lat.h"
#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_REL_DIR_TO_LON_LAT(oskar_convert_relative_directions_to_lon_lat_2d_f, 0, float)
OSKAR_CONVERT_REL_DIR_TO_LON_LAT(oskar_convert_relative_directions_to_lon_lat_3d_f, 1, float)
OSKAR_CONVERT_REL_DIR_TO_LON_LAT(oskar_convert_relative_directions_to_lon_lat_2d_d, 0, double)
OSKAR_CONVERT_REL_DIR_TO_LON_LAT(oskar_convert_relative_directions_to_lon_lat_3d_d, 1, double)

void oskar_convert_relative_directions_to_lon_lat(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        double lon0_rad, double lat0_rad,
        oskar_Mem* lon_rad, oskar_Mem* lat_rad, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(l);
    const int location = oskar_mem_location(l);
    const int is_3d = (n != 0);
    const double sin_lat0 = sin(lat0_rad);
    const double cos_lat0 = cos(lat0_rad);
    const float lon0_rad_f = (float) lon0_rad;
    const float sin_lat0_f = (float) sin_lat0;
    const float cos_lat0_f = (float) cos_lat0;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            if (is_3d)
            {
                oskar_convert_relative_directions_to_lon_lat_3d_f(num_points,
                        oskar_mem_float_const(l, status),
                        oskar_mem_float_const(m, status),
                        oskar_mem_float_const(n, status),
                        lon0_rad_f, cos_lat0_f, sin_lat0_f,
                        oskar_mem_float(lon_rad, status),
                        oskar_mem_float(lat_rad, status));
            }
            else
            {
                oskar_convert_relative_directions_to_lon_lat_2d_f(num_points,
                        oskar_mem_float_const(l, status),
                        oskar_mem_float_const(m, status), 0,
                        lon0_rad_f, cos_lat0_f, sin_lat0_f,
                        oskar_mem_float(lon_rad, status),
                        oskar_mem_float(lat_rad, status));
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            if (is_3d)
            {
                oskar_convert_relative_directions_to_lon_lat_3d_d(num_points,
                        oskar_mem_double_const(l, status),
                        oskar_mem_double_const(m, status),
                        oskar_mem_double_const(n, status),
                        lon0_rad, cos_lat0, sin_lat0,
                        oskar_mem_double(lon_rad, status),
                        oskar_mem_double(lat_rad, status));
            }
            else
            {
                oskar_convert_relative_directions_to_lon_lat_2d_d(num_points,
                        oskar_mem_double_const(l, status),
                        oskar_mem_double_const(m, status), 0,
                        lon0_rad, cos_lat0, sin_lat0,
                        oskar_mem_double(lon_rad, status),
                        oskar_mem_double(lat_rad, status));
            }
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
        const void* nullp = 0;
        const int is_dbl = oskar_mem_is_double(lon_rad);
        if (type == OSKAR_SINGLE)
        {
            k = is_3d ?
                    "convert_relative_directions_to_lon_lat_3d_float" :
                    "convert_relative_directions_to_lon_lat_2d_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = is_3d ?
                    "convert_relative_directions_to_lon_lat_3d_double" :
                    "convert_relative_directions_to_lon_lat_2d_double";
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
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {PTR_SZ, (is_3d ? oskar_mem_buffer_const(n) : &nullp)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&lon0_rad : (const void*)&lon0_rad_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&cos_lat0 : (const void*)&cos_lat0_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&sin_lat0 : (const void*)&sin_lat0_f},
                {PTR_SZ, oskar_mem_buffer(lon_rad)},
                {PTR_SZ, oskar_mem_buffer(lat_rad)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
