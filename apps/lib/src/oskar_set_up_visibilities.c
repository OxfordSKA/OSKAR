/*
 * Copyright (c) 2011-2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include "apps/lib/oskar_set_up_visibilities.h"
#include <oskar_convert_ecef_to_baseline_uvw.h>
#include <oskar_cmath.h>

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Vis* oskar_set_up_visibilities(const oskar_Settings* settings,
        const oskar_Telescope* tel, int* status)
{
    int num_stations, num_channels, precision, vis_type;
    double rad2deg = 180.0/M_PI;
    oskar_Vis* vis = 0;

    /* Check all inputs. */
    if (!settings || !tel || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Initialise the global visibility structure on the CPU. */
    num_stations = oskar_telescope_num_stations(tel);
    num_channels = settings->obs.num_channels;
    precision = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    vis_type = precision | OSKAR_COMPLEX;
    if (oskar_telescope_pol_mode(tel) == OSKAR_POL_MODE_FULL)
        vis_type |= OSKAR_MATRIX;
    vis = oskar_vis_create(vis_type, OSKAR_CPU,
            num_channels, settings->obs.num_time_steps, num_stations, status);

    /* Add meta-data. */
    oskar_vis_set_freq_start_hz(vis, settings->obs.start_frequency_hz);
    oskar_vis_set_freq_inc_hz(vis, settings->obs.frequency_inc_hz);
    oskar_vis_set_time_start_mjd_utc(vis, settings->obs.start_mjd_utc);
    oskar_vis_set_time_inc_sec(vis, settings->obs.dt_dump_days * 86400.0);
    oskar_vis_set_time_average_sec(vis,
            settings->interferometer.time_average_sec);
    oskar_vis_set_channel_bandwidth_hz(vis,
            settings->interferometer.channel_bandwidth_hz);
    oskar_vis_set_phase_centre(vis,
            settings->obs.phase_centre_lon_rad[0] * rad2deg,
            settings->obs.phase_centre_lat_rad[0] * rad2deg);
    oskar_vis_set_telescope_position(vis,
            oskar_telescope_lon_rad(tel) * rad2deg,
            oskar_telescope_lat_rad(tel) * rad2deg,
            oskar_telescope_alt_metres(tel));

    /* Add settings file path. */
    oskar_mem_append_raw(oskar_vis_settings_path(vis),
            settings->settings_path, OSKAR_CHAR, OSKAR_CPU,
            1 + strlen(settings->settings_path), status);

    /* Add telescope model path. */
    oskar_mem_append_raw(oskar_vis_telescope_path(vis),
            settings->telescope.input_directory, OSKAR_CHAR, OSKAR_CPU,
            1 + strlen(settings->telescope.input_directory), status);

    /* Add settings file contents. */
    {
        oskar_Mem* temp;
        temp = oskar_mem_read_binary_raw(settings->settings_path,
                OSKAR_CHAR, OSKAR_CPU, status);
        oskar_mem_copy(oskar_vis_settings(vis), temp, status);
        oskar_mem_free(temp, status);
    }

    /* Copy station coordinates from telescope model. */
    oskar_mem_copy(oskar_vis_station_x_offset_ecef_metres(vis),
            oskar_telescope_station_true_x_offset_ecef_metres_const(tel),
            status);
    oskar_mem_copy(oskar_vis_station_y_offset_ecef_metres(vis),
            oskar_telescope_station_true_y_offset_ecef_metres_const(tel),
            status);
    oskar_mem_copy(oskar_vis_station_z_offset_ecef_metres(vis),
            oskar_telescope_station_true_z_offset_ecef_metres_const(tel),
            status);

    /* Compute baseline u,v,w coordinates for simulation. */
    {
        oskar_Mem *work_uvw;
        work_uvw = oskar_mem_create(oskar_mem_type_precision(vis_type),
                OSKAR_CPU, 3 * num_stations, status);
        oskar_convert_ecef_to_baseline_uvw(
                oskar_telescope_num_stations(tel),
                oskar_telescope_station_true_x_offset_ecef_metres_const(tel),
                oskar_telescope_station_true_y_offset_ecef_metres_const(tel),
                oskar_telescope_station_true_z_offset_ecef_metres_const(tel),
                oskar_telescope_phase_centre_ra_rad(tel),
                oskar_telescope_phase_centre_dec_rad(tel),
                settings->obs.num_time_steps, settings->obs.start_mjd_utc,
                settings->obs.dt_dump_days, 0,
                oskar_vis_baseline_uu_metres(vis),
                oskar_vis_baseline_vv_metres(vis),
                oskar_vis_baseline_ww_metres(vis), work_uvw, status);
        oskar_mem_free(work_uvw, status);
    }

    return vis;
}

#ifdef __cplusplus
}
#endif
