/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <stdlib.h>
#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288
#endif

#ifdef __cplusplus
extern "C" {
#endif

oskar_Vis* oskar_set_up_visibilities(const oskar_Settings* settings,
        const oskar_Telescope* telescope, int type, int* status)
{
    int num_stations, num_channels, i;
    double rad2deg = 180.0/M_PI;
    oskar_Vis* vis = 0;

    /* Check all inputs. */
    if (!settings || !telescope || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check the type. */
    if (!oskar_mem_type_is_complex(type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Initialise the global visibility structure on the CPU. */
    num_stations = oskar_telescope_num_stations(telescope);
    num_channels = settings->obs.num_channels;
    vis = oskar_vis_create(type, OSKAR_LOCATION_CPU,
            num_channels, settings->obs.num_time_steps, num_stations, status);

    /* Add meta-data. */
    oskar_vis_set_freq_start_hz(vis, settings->obs.start_frequency_hz);
    oskar_vis_set_freq_inc_hz(vis, settings->obs.frequency_inc_hz);
    oskar_vis_set_time_start_mjd_utc(vis, settings->obs.start_mjd_utc);
    oskar_vis_set_time_inc_seconds(vis, settings->obs.dt_dump_days * 86400.0);
    oskar_vis_set_time_int_seconds(vis, settings->interferometer.time_average_sec);
    oskar_vis_set_channel_bandwidth_hz(vis, settings->interferometer.channel_bandwidth_hz);
    oskar_vis_set_phase_centre(vis, settings->obs.ra0_rad[0] * rad2deg,
            settings->obs.dec0_rad[0] * rad2deg);
    oskar_vis_set_telescope_position(vis,
            oskar_telescope_longitude_rad(telescope) * rad2deg,
            oskar_telescope_latitude_rad(telescope) * rad2deg);

    /* Add settings file path. */
    oskar_mem_append_raw(oskar_vis_settings_path(vis),
            settings->settings_path, OSKAR_CHAR, OSKAR_LOCATION_CPU,
            1 + strlen(settings->settings_path), status);

    /* Add telescope model path. */
    oskar_mem_append_raw(oskar_vis_telescope_path(vis),
            settings->telescope.input_directory, OSKAR_CHAR,
            OSKAR_LOCATION_CPU, 1 + strlen(settings->telescope.input_directory),
            status);

    /* Copy station coordinates from telescope model. */
    oskar_mem_copy(oskar_vis_station_x_metres(vis),
            oskar_telescope_station_x_const(telescope), status);
    oskar_mem_copy(oskar_vis_station_y_metres(vis),
            oskar_telescope_station_y_const(telescope), status);
    oskar_mem_copy(oskar_vis_station_z_metres(vis),
            oskar_telescope_station_z_const(telescope), status);

    /* Compute baseline u,v,w coordinates for simulation. */
    {
        oskar_Mem *work_uvw;
        work_uvw = oskar_mem_create(oskar_mem_type_precision(type),
                OSKAR_LOCATION_CPU, 3 * num_stations, status);
        oskar_convert_ecef_to_baseline_uvw(oskar_vis_baseline_uu_metres(vis),
                oskar_vis_baseline_vv_metres(vis),
                oskar_vis_baseline_ww_metres(vis),
                oskar_telescope_num_stations(telescope),
                oskar_telescope_station_x_const(telescope),
                oskar_telescope_station_y_const(telescope),
                oskar_telescope_station_z_const(telescope),
                oskar_telescope_ra0_rad(telescope),
                oskar_telescope_dec0_rad(telescope),
                settings->obs.num_time_steps, settings->obs.start_mjd_utc,
                settings->obs.dt_dump_days, work_uvw, status);
        oskar_mem_free(work_uvw, status);
    }

    /* Copy station lon/lat and nominal receptor orientations */
    if (settings->sim.double_precision)
    {
        double* lon = oskar_mem_double(oskar_vis_station_lon_deg(vis), status);
        double* lat = oskar_mem_double(oskar_vis_station_lat_deg(vis), status);
        double* orientation_x = oskar_mem_double(
                oskar_vis_station_orientation_x_deg(vis), status);
        double* orientation_y = oskar_mem_double(
                oskar_vis_station_orientation_y_deg(vis), status);
        for (i = 0; i < num_stations; ++i)
        {
            const oskar_Station* station;
            station = oskar_telescope_station_const(telescope, i);
            lon[i] = oskar_station_longitude_rad(station) * rad2deg;
            lat[i] = oskar_station_latitude_rad(station) * rad2deg;
            orientation_x[i] =
                    oskar_station_element_orientation_x_rad(station) * rad2deg;
            orientation_y[i] =
                    oskar_station_element_orientation_y_rad(station) * rad2deg;
        }
    }
    else
    {
        float* lon = oskar_mem_float(oskar_vis_station_lon_deg(vis), status);
        float* lat = oskar_mem_float(oskar_vis_station_lat_deg(vis), status);
        float* orientation_x = oskar_mem_float(
                oskar_vis_station_orientation_x_deg(vis), status);
        float* orientation_y = oskar_mem_float(
                oskar_vis_station_orientation_y_deg(vis), status);
        for (i = 0; i < num_stations; ++i)
        {
            const oskar_Station* station;
            station = oskar_telescope_station_const(telescope, i);
            lon[i] = oskar_station_longitude_rad(station) * rad2deg;
            lat[i] = oskar_station_latitude_rad(station) * rad2deg;
            orientation_x[i] =
                    oskar_station_element_orientation_x_rad(station) * rad2deg;
            orientation_y[i] =
                    oskar_station_element_orientation_y_rad(station) * rad2deg;
        }
    }
    return vis;
}

#ifdef __cplusplus
}
#endif
