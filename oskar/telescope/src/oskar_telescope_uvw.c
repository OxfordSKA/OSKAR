/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"
#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "convert/oskar_convert_ecef_to_station_uvw.h"
#include "convert/oskar_convert_mjd_to_gast_fast.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_uvw(
        const oskar_Telescope* tel,
        int use_true_coords,
        int ignore_w_components,
        int num_times,
        double time_ref_mjd_utc,
        double time_inc_days,
        int start_time_index,
        oskar_Mem* u,
        oskar_Mem* v,
        oskar_Mem* w,
        oskar_Mem* uu,
        oskar_Mem* vv,
        oskar_Mem* ww,
        int* status)
{
    int i = 0;
    const oskar_Mem *xyz[3];
    if (*status) return;
    const int num_stations = oskar_telescope_num_stations(tel);
    const int num_baselines = num_stations * (num_stations - 1) / 2;
    const int total_baselines = num_baselines * num_times;
    const int total_stations = num_stations * num_times;
    const int coord_type = oskar_telescope_phase_centre_coord_type(tel);
    double lon0_rad = oskar_telescope_phase_centre_longitude_rad(tel);
    double lat0_rad = oskar_telescope_phase_centre_latitude_rad(tel);
    for (i = 0; i < 3; ++i)
    {
        if (coord_type == OSKAR_COORDS_AZEL)
        {
            xyz[i] = use_true_coords ?
                    oskar_telescope_station_true_enu_metres_const(tel, i) :
                    oskar_telescope_station_measured_enu_metres_const(tel, i);
        }
        else
        {
            xyz[i] = use_true_coords ?
                    oskar_telescope_station_true_offset_ecef_metres_const(tel, i) :
                    oskar_telescope_station_measured_offset_ecef_metres_const(tel, i);
        }
    }
    oskar_mem_ensure(u, total_stations, status);
    oskar_mem_ensure(v, total_stations, status);
    oskar_mem_ensure(w, total_stations, status);
    if (uu) oskar_mem_ensure(uu, total_baselines, status);
    if (vv) oskar_mem_ensure(vv, total_baselines, status);
    if (ww) oskar_mem_ensure(ww, total_baselines, status);
    if (*status) return;
    for (i = 0; i < num_times; ++i)
    {
        if (coord_type == OSKAR_COORDS_AZEL)
        {
            oskar_mem_copy_contents(u, xyz[0],
                    i * num_stations, 0, num_stations, status);
            oskar_mem_copy_contents(v, xyz[1],
                    i * num_stations, 0, num_stations, status);
            if (ignore_w_components)
            {
                oskar_mem_clear_contents(w, status);
            }
            else
            {
                oskar_mem_copy_contents(w, xyz[2],
                        i * num_stations, 0, num_stations, status);
            }
        }
        else
        {
            const double t_dump = time_ref_mjd_utc +
                    time_inc_days * ((i + 0.5) + start_time_index);
            const double gast = oskar_convert_mjd_to_gast_fast(t_dump);
            oskar_convert_ecef_to_station_uvw(
                    num_stations, xyz[0], xyz[1], xyz[2],
                    lon0_rad, lat0_rad, gast, ignore_w_components,
                    i * num_stations, u, v, w, status);
        }
        if (uu && vv && ww)
        {
            oskar_convert_station_uvw_to_baseline_uvw(num_stations,
                    i * num_stations, u, v, w,
                    i * num_baselines, uu, vv, ww, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
