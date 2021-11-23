/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

#define CONVERT_STATION_TO_BASELINE(NAME, FP) static void NAME(\
        const int num_stations,\
        const int offset_in, const FP *u, const FP *v, const FP *w,\
        const int offset_out, FP *uu, FP *vv, FP *ww)\
{\
    int s1, s2, b;\
    for (s1 = 0, b = 0; s1 < num_stations; ++s1) {\
        const int i1 = s1 + offset_in;\
        for (s2 = s1 + 1; s2 < num_stations; ++s2, ++b) {\
            const int i2 = s2 + offset_in;\
            const int out = b + offset_out;\
            uu[out] = u[i2] - u[i1];\
            vv[out] = v[i2] - v[i1];\
            ww[out] = w[i2] - w[i1];\
        }\
    }\
}

CONVERT_STATION_TO_BASELINE(convert_station_uvw_to_baseline_uvw_float, float)
CONVERT_STATION_TO_BASELINE(convert_station_uvw_to_baseline_uvw_double, double)

void oskar_convert_station_uvw_to_baseline_uvw(int num_stations, int offset_in,
        const oskar_Mem* u, const oskar_Mem* v, const oskar_Mem* w,
        int offset_out, oskar_Mem* uu, oskar_Mem* vv, oskar_Mem* ww,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_type(u);
    const int location = oskar_mem_location(u);
    if (location == OSKAR_CPU)
    {
        if (type == OSKAR_SINGLE)
        {
            convert_station_uvw_to_baseline_uvw_float(num_stations, offset_in,
                    oskar_mem_float_const(u, status),
                    oskar_mem_float_const(v, status),
                    oskar_mem_float_const(w, status), offset_out,
                    oskar_mem_float(uu, status),
                    oskar_mem_float(vv, status),
                    oskar_mem_float(ww, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            convert_station_uvw_to_baseline_uvw_double(num_stations, offset_in,
                    oskar_mem_double_const(u, status),
                    oskar_mem_double_const(v, status),
                    oskar_mem_double_const(w, status), offset_out,
                    oskar_mem_double(uu, status),
                    oskar_mem_double(vv, status),
                    oskar_mem_double(ww, status));
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        size_t local_size[] = {32, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        if (type == OSKAR_SINGLE)
        {
            k = "convert_station_uvw_to_baseline_uvw_float";
        }
        else if (type == OSKAR_DOUBLE)
        {
            k = "convert_station_uvw_to_baseline_uvw_double";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = num_stations * local_size[0];
        const oskar_Arg args[] = {
                {INT_SZ, &num_stations},
                {INT_SZ, &offset_in},
                {PTR_SZ, oskar_mem_buffer_const(u)},
                {PTR_SZ, oskar_mem_buffer_const(v)},
                {PTR_SZ, oskar_mem_buffer_const(w)},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(uu)},
                {PTR_SZ, oskar_mem_buffer(vv)},
                {PTR_SZ, oskar_mem_buffer(ww)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
