/*
 * Copyright (c) 2020-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_apparent_ra_dec_to_enu_directions.h"
#include "convert/oskar_convert_apparent_ra_dec_to_enu_directions.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_RA_DEC_TO_ENU_DIR(oskar_convert_apparent_ra_dec_to_enu_directions_float, float)
OSKAR_CONVERT_RA_DEC_TO_ENU_DIR(oskar_convert_apparent_ra_dec_to_enu_directions_double, double)

void oskar_convert_apparent_ra_dec_to_enu_directions(
        int num_points, const oskar_Mem* ra_rad, const oskar_Mem* dec_rad,
        double lst_rad, double latitude_rad, int offset_out,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(ra_rad);
    const int location = oskar_mem_location(ra_rad);
    const double sin_lat = sin(latitude_rad);
    const double cos_lat = cos(latitude_rad);
    const float sin_lat_f = (float)sin_lat;
    const float cos_lat_f = (float)cos_lat;
    const float lst_rad_f = (float)lst_rad;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            oskar_convert_apparent_ra_dec_to_enu_directions_float(num_points,
                    oskar_mem_float_const(ra_rad, status),
                    oskar_mem_float_const(dec_rad, status),
                    lst_rad_f, sin_lat_f, cos_lat_f, offset_out,
                    oskar_mem_float(x, status),
                    oskar_mem_float(y, status),
                    oskar_mem_float(z, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            oskar_convert_apparent_ra_dec_to_enu_directions_double(num_points,
                    oskar_mem_double_const(ra_rad, status),
                    oskar_mem_double_const(dec_rad, status),
                    lst_rad, sin_lat, cos_lat, offset_out,
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
        const int is_dbl = (type == OSKAR_DOUBLE);
        if (type == OSKAR_SINGLE)
        {
            k = "convert_apparent_ra_dec_to_enu_directions_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_apparent_ra_dec_to_enu_directions_double";
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
                {PTR_SZ, oskar_mem_buffer_const(ra_rad)},
                {PTR_SZ, oskar_mem_buffer_const(dec_rad)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&lst_rad : (const void*)&lst_rad_f},
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
