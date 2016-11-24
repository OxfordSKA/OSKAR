/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "vis/private_vis.h"
#include "vis/oskar_vis.h"

#ifdef __cplusplus
extern "C" {
#endif


int oskar_vis_location(const oskar_Vis* vis)
{
    return oskar_mem_location(vis->amplitude);
}

oskar_Mem* oskar_vis_settings_path(oskar_Vis* vis)
{
    return vis->settings_path;
}

const oskar_Mem* oskar_vis_settings_path_const(const oskar_Vis* vis)
{
    return vis->settings_path;
}

oskar_Mem* oskar_vis_telescope_path(oskar_Vis* vis)
{
    return vis->telescope_path;
}

const oskar_Mem* oskar_vis_telescope_path_const(const oskar_Vis* vis)
{
    return vis->telescope_path;
}

oskar_Mem* oskar_vis_settings(oskar_Vis* vis)
{
    return vis->settings;
}

const oskar_Mem* oskar_vis_settings_const(const oskar_Vis* vis)
{
    return vis->settings;
}

int oskar_vis_num_channels(const oskar_Vis* vis)
{
    return vis->num_channels;
}

int oskar_vis_num_times(const oskar_Vis* vis)
{
    return vis->num_times;
}

int oskar_vis_num_stations(const oskar_Vis* vis)
{
    return vis->num_stations;
}

int oskar_vis_num_baselines(const oskar_Vis* vis)
{
    return vis->num_baselines;
}

int oskar_vis_num_pols(const oskar_Vis* vis)
{
    return oskar_mem_is_matrix(vis->amplitude) ? 4 : 1;
}

double oskar_vis_freq_start_hz(const oskar_Vis* vis)
{
    return vis->freq_start_hz;
}

double oskar_vis_freq_inc_hz(const oskar_Vis* vis)
{
    return vis->freq_inc_hz;
}

double oskar_vis_channel_bandwidth_hz(const oskar_Vis* vis)
{
    return vis->channel_bandwidth_hz;
}

double oskar_vis_time_start_mjd_utc(const oskar_Vis* vis)
{
    return vis->time_start_mjd_utc;
}

double oskar_vis_time_inc_sec(const oskar_Vis* vis)
{
    return vis->time_inc_sec;
}

double oskar_vis_time_average_sec(const oskar_Vis* vis)
{
    return vis->time_average_sec;
}

double oskar_vis_phase_centre_ra_deg(const oskar_Vis* vis)
{
    return vis->phase_centre_ra_deg;
}

double oskar_vis_phase_centre_dec_deg(const oskar_Vis* vis)
{
    return vis->phase_centre_dec_deg;
}

double oskar_vis_telescope_lon_deg(const oskar_Vis* vis)
{
    return vis->telescope_lon_deg;
}

double oskar_vis_telescope_lat_deg(const oskar_Vis* vis)
{
    return vis->telescope_lat_deg;
}

double oskar_vis_telescope_alt_metres(const oskar_Vis* vis)
{
    return vis->telescope_alt_metres;
}

oskar_Mem* oskar_vis_station_x_offset_ecef_metres(oskar_Vis* vis)
{
    return vis->station_x_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_station_x_offset_ecef_metres_const(
        const oskar_Vis* vis)
{
    return vis->station_x_offset_ecef_metres;
}

oskar_Mem* oskar_vis_station_y_offset_ecef_metres(oskar_Vis* vis)
{
    return vis->station_y_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_station_y_offset_ecef_metres_const(
        const oskar_Vis* vis)
{
    return vis->station_y_offset_ecef_metres;
}

oskar_Mem* oskar_vis_station_z_offset_ecef_metres(oskar_Vis* vis)
{
    return vis->station_z_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_station_z_offset_ecef_metres_const(
        const oskar_Vis* vis)
{
    return vis->station_z_offset_ecef_metres;
}

oskar_Mem* oskar_vis_baseline_uu_metres(oskar_Vis* vis)
{
    return vis->baseline_uu_metres;
}

const oskar_Mem* oskar_vis_baseline_uu_metres_const(const oskar_Vis* vis)
{
    return vis->baseline_uu_metres;
}

oskar_Mem* oskar_vis_baseline_vv_metres(oskar_Vis* vis)
{
    return vis->baseline_vv_metres;
}

const oskar_Mem* oskar_vis_baseline_vv_metres_const(const oskar_Vis* vis)
{
    return vis->baseline_vv_metres;
}

oskar_Mem* oskar_vis_baseline_ww_metres(oskar_Vis* vis)
{
    return vis->baseline_ww_metres;
}

const oskar_Mem* oskar_vis_baseline_ww_metres_const(const oskar_Vis* vis)
{
    return vis->baseline_ww_metres;
}

oskar_Mem* oskar_vis_amplitude(oskar_Vis* vis)
{
    return vis->amplitude;
}

const oskar_Mem* oskar_vis_amplitude_const(const oskar_Vis* vis)
{
    return vis->amplitude;
}

void oskar_vis_set_freq_start_hz(oskar_Vis* vis, double value)
{
    vis->freq_start_hz = value;
}

void oskar_vis_set_freq_inc_hz(oskar_Vis* vis, double value)
{
    vis->freq_inc_hz = value;
}

void oskar_vis_set_channel_bandwidth_hz(oskar_Vis* vis, double value)
{
    vis->channel_bandwidth_hz = value;
}

void oskar_vis_set_time_start_mjd_utc(oskar_Vis* vis, double value)
{
    vis->time_start_mjd_utc = value;
}

void oskar_vis_set_time_inc_sec(oskar_Vis* vis, double value)
{
    vis->time_inc_sec = value;
}

void oskar_vis_set_time_average_sec(oskar_Vis* vis, double value)
{
    vis->time_average_sec = value;
}

void oskar_vis_set_phase_centre(oskar_Vis* vis, double ra_deg, double dec_deg)
{
    vis->phase_centre_ra_deg = ra_deg;
    vis->phase_centre_dec_deg = dec_deg;
}

void oskar_vis_set_telescope_position(oskar_Vis* vis, double lon_deg,
        double lat_deg, double alt_metres)
{
    vis->telescope_lon_deg = lon_deg;
    vis->telescope_lat_deg = lat_deg;
    vis->telescope_alt_metres = alt_metres;
}

#ifdef __cplusplus
}
#endif
