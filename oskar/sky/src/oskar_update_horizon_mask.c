/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/define_update_horizon_mask.h"
#include "sky/oskar_update_horizon_mask.h"
#include "utility/oskar_device.h"
#include "utility/oskar_kernel_macros.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_UPDATE_HORIZON_MASK(update_horizon_mask_float, float)
OSKAR_UPDATE_HORIZON_MASK(update_horizon_mask_double, double)

void oskar_update_horizon_mask(int num_sources, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ha0_rad,
        const double dec0_rad, const double lat_rad, oskar_Mem* mask,
        int* status)
{
    if (*status) return;
    const int type = oskar_mem_precision(l);
    const int location = oskar_mem_location(mask);
    const double cos_ha0  = cos(ha0_rad);
    const double sin_dec0 = sin(dec0_rad);
    const double cos_dec0 = cos(dec0_rad);
    const double sin_lat  = sin(lat_rad);
    const double cos_lat  = cos(lat_rad);
    const double ll = cos_lat * sin(ha0_rad);
    const double mm = sin_lat * cos_dec0 - cos_lat * cos_ha0 * sin_dec0;
    const double nn = sin_lat * sin_dec0 + cos_lat * cos_ha0 * cos_dec0;
    const float ll_ = (float) ll, mm_ = (float) mm, nn_ = (float) nn;
    if (location == OSKAR_CPU)
    {
#if 0
        /* About 10 times slower! */
        if (type == OSKAR_SINGLE)
            update_horizon_mask_float(num_sources,
                    oskar_mem_float_const(ra_rad, status),
                    oskar_mem_float_const(dec_rad, status),
                    lst_rad, cos_lat, sin_lat,
                    oskar_mem_int(mask, status));
        else if (type == OSKAR_DOUBLE)
            update_horizon_mask_double(num_sources,
                    oskar_mem_double_const(ra_rad, status),
                    oskar_mem_double_const(dec_rad, status),
                    lst_rad, cos_lat, sin_lat,
                    oskar_mem_int(mask, status));
#endif
        if (type == OSKAR_SINGLE)
        {
            update_horizon_mask_float(num_sources,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    oskar_mem_float_const(n, status), ll_, mm_, nn_,
                    oskar_mem_int(mask, status));
        }
        else if (type == OSKAR_DOUBLE)
        {
            update_horizon_mask_double(num_sources,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    oskar_mem_double_const(n, status), ll_, mm_, nn_,
                    oskar_mem_int(mask, status));
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
        if (type == OSKAR_DOUBLE)
        {
            k = "update_horizon_mask_double";
        }
        else if (type == OSKAR_SINGLE)
        {
            k = "update_horizon_mask_float";
        }
        else
        {
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_sources, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &num_sources},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {PTR_SZ, oskar_mem_buffer_const(n)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&ll : (const void*)&ll_},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&mm : (const void*)&mm_},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&nn : (const void*)&nn_},
                {PTR_SZ, oskar_mem_buffer(mask)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
