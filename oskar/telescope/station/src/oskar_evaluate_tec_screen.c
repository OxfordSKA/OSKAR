/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_evaluate_tec_screen.h"
#include "telescope/station/define_evaluate_tec_screen.h"
#include "utility/oskar_kernel_macros.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EVALUATE_TEC_SCREEN(evaluate_tec_screen_float, float, float2)
OSKAR_EVALUATE_TEC_SCREEN(evaluate_tec_screen_double, double, double2)

void oskar_evaluate_tec_screen(
        int isoplanatic,
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        double station_u_m,
        double station_v_m,
        double frequency_hz,
        double screen_height_m,
        double screen_pixel_size_m,
        int screen_num_pixels_x,
        int screen_num_pixels_y,
        const oskar_Mem* tec_screen,
        int offset_out,
        oskar_Mem* out,
        int* status)
{
    if (*status) return;
    const double inv_pixel_size_m = 1.0 / screen_pixel_size_m;
    const double inv_freq_hz = 1.0 / frequency_hz;
    const float inv_freq_hz_f = (float) inv_freq_hz;
    const float inv_pixel_size_m_f = (float) inv_pixel_size_m;
    const float station_u_f = (float) station_u_m;
    const float station_v_f = (float) station_v_m;
    const float screen_height_m_f = (float) screen_height_m;
    const int location = oskar_mem_location(tec_screen);
    if (location == OSKAR_CPU)
    {
        if (oskar_mem_type(tec_screen) == OSKAR_SINGLE)
        {
            evaluate_tec_screen_float(isoplanatic, num_points,
                    oskar_mem_float_const(l, status),
                    oskar_mem_float_const(m, status),
                    station_u_f, station_v_f, inv_freq_hz_f,
                    screen_height_m_f, inv_pixel_size_m_f,
                    screen_num_pixels_x, screen_num_pixels_y,
                    oskar_mem_float_const(tec_screen, status), offset_out,
                    oskar_mem_float2(out, status));
        }
        else
        {
            evaluate_tec_screen_double(isoplanatic, num_points,
                    oskar_mem_double_const(l, status),
                    oskar_mem_double_const(m, status),
                    station_u_m, station_v_m, inv_freq_hz,
                    screen_height_m, inv_pixel_size_m,
                    screen_num_pixels_x, screen_num_pixels_y,
                    oskar_mem_double_const(tec_screen, status), offset_out,
                    oskar_mem_double2(out, status));
        }
    }
    else
    {
        size_t local_size[] = {256, 1, 1}, global_size[] = {1, 1, 1};
        const char* k = 0;
        const int is_dbl = oskar_mem_is_double(tec_screen);
        switch (oskar_mem_type(tec_screen))
        {
        case OSKAR_SINGLE:
            k = "evaluate_tec_screen_float"; break;
        case OSKAR_DOUBLE:
            k = "evaluate_tec_screen_double"; break;
        default:
            *status = OSKAR_ERR_BAD_DATA_TYPE;
            return;
        }
        oskar_device_check_local_size(location, 0, local_size);
        global_size[0] = oskar_device_global_size(
                (size_t) num_points, local_size[0]);
        const oskar_Arg args[] = {
                {INT_SZ, &isoplanatic},
                {INT_SZ, &num_points},
                {PTR_SZ, oskar_mem_buffer_const(l)},
                {PTR_SZ, oskar_mem_buffer_const(m)},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&station_u_m : (const void*)&station_u_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&station_v_m : (const void*)&station_v_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&inv_freq_hz :
                        (const void*)&inv_freq_hz_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&screen_height_m :
                        (const void*)&screen_height_m_f},
                {is_dbl ? DBL_SZ : FLT_SZ, is_dbl ?
                        (const void*)&inv_pixel_size_m :
                        (const void*)&inv_pixel_size_m_f},
                {INT_SZ, &screen_num_pixels_x},
                {INT_SZ, &screen_num_pixels_y},
                {PTR_SZ, oskar_mem_buffer_const(tec_screen)},
                {INT_SZ, &offset_out},
                {PTR_SZ, oskar_mem_buffer(out)}
        };
        oskar_device_launch_kernel(k, location, 1, local_size, global_size,
                sizeof(args) / sizeof(oskar_Arg), args, 0, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
