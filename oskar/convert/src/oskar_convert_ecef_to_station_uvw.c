/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/define_convert_ecef_to_station_uvw.h"
#include "convert/oskar_convert_ecef_to_station_uvw.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_CONVERT_ECEF_TO_STATION_UVW(convert_ecef_to_station_uvw_float, float)
OSKAR_CONVERT_ECEF_TO_STATION_UVW(convert_ecef_to_station_uvw_double, double)

void oskar_convert_ecef_to_station_uvw(int num_stations,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ra0_rad, double dec0_rad, double gast, int ignore_w_components,
        int offset_out, oskar_Mem* u, oskar_Mem* v, oskar_Mem* w, int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(x);
    const int location = oskar_mem_location(x);
    if (oskar_mem_location(y) != location ||
            oskar_mem_location(z) != location ||
            oskar_mem_location(u) != location ||
            oskar_mem_location(v) != location ||
            oskar_mem_location(w) != location)
    {
        *status = OSKAR_ERR_LOCATION_MISMATCH;
        return;
    }
    if (oskar_mem_type(y) != type || oskar_mem_type(z) != type ||
            oskar_mem_type(u) != type || oskar_mem_type(v) != type ||
            oskar_mem_type(w) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    oskar_mem_ensure(u, num_stations, status);
    oskar_mem_ensure(v, num_stations, status);
    oskar_mem_ensure(w, num_stations, status);
    const double ha0_rad = gast - ra0_rad;
    const double sin_ha0  = sin(ha0_rad);
    const double cos_ha0  = cos(ha0_rad);
    const double sin_dec0 = sin(dec0_rad);
    const double cos_dec0 = cos(dec0_rad);
    const float sin_ha0_f = (float) sin_ha0;
    const float cos_ha0_f = (float) cos_ha0;
    const float sin_dec0_f = (float) sin_dec0;
    const float cos_dec0_f = (float) cos_dec0;
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
            convert_ecef_to_station_uvw_float(num_stations,
                    oskar_mem_float_const(x, status),
                    oskar_mem_float_const(y, status),
                    oskar_mem_float_const(z, status),
                    sin_ha0_f, cos_ha0_f, sin_dec0_f, cos_dec0_f,
                    ignore_w_components, offset_out,
                    oskar_mem_float(u, status),
                    oskar_mem_float(v, status),
                    oskar_mem_float(w, status));
        else if (type == OSKAR_DOUBLE)
            convert_ecef_to_station_uvw_double(num_stations,
                    oskar_mem_double_const(x, status),
                    oskar_mem_double_const(y, status),
                    oskar_mem_double_const(z, status),
                    sin_ha0, cos_ha0, sin_dec0, cos_dec0,
                    ignore_w_components, offset_out,
                    oskar_mem_double(u, status),
                    oskar_mem_double(v, status),
                    oskar_mem_double(w, status));
        else
            *status = OSKAR_ERR_BAD_DATA_TYPE;
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(x);
        if (type == OSKAR_SINGLE)
            k = "convert_ecef_to_station_uvw_float";
        else if (type == OSKAR_DOUBLE)
            k = "convert_ecef_to_station_uvw_double";
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_stations, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_stations},
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
                {INT_SZ, &ignore_w_components},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(u)},
                {PTR_SZ, oskar_mem_buffer(v)},
                {PTR_SZ, oskar_mem_buffer(w)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
