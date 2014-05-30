/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <private_vis.h>
#include <oskar_vis.h>
#include <oskar_mem.h>
#include <math.h>
#include <oskar_convert_offset_ecef_to_ecef.h>
#include <oskar_convert_ecef_to_enu.h>

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

int oskar_vis_num_polarisations(const oskar_Vis* vis)
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

double oskar_vis_time_inc_seconds(const oskar_Vis* vis)
{
    return vis->time_inc_seconds;
}

double oskar_vis_time_int_seconds(const oskar_Vis* vis)
{
    return vis->time_int_seconds;
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

oskar_Mem* oskar_vis_station_x_metres(oskar_Vis* vis)
{
    return vis->x_metres;
}

const oskar_Mem* oskar_vis_station_x_metres_const(const oskar_Vis* vis)
{
    return vis->x_metres;
}

oskar_Mem* oskar_vis_station_y_metres(oskar_Vis* vis)
{
    return vis->y_metres;
}

const oskar_Mem* oskar_vis_station_y_metres_const(const oskar_Vis* vis)
{
    return vis->y_metres;
}

oskar_Mem* oskar_vis_station_z_metres(oskar_Vis* vis)
{
    return vis->z_metres;
}

const oskar_Mem* oskar_vis_station_z_metres_const(const oskar_Vis* vis)
{
    return vis->z_metres;
}

inline static void station_xyz_to_horizon(const oskar_Vis* vis, oskar_Mem* x,
        oskar_Mem* y, oskar_Mem* z)
{
    int status = OSKAR_SUCCESS;
    oskar_Mem* x_ = oskar_mem_convert_precision(vis->x_metres, OSKAR_DOUBLE, &status);
    oskar_Mem* y_ = oskar_mem_convert_precision(vis->y_metres, OSKAR_DOUBLE, &status);
    oskar_Mem* z_ = oskar_mem_convert_precision(vis->z_metres, OSKAR_DOUBLE, &status);

    int n = vis->num_stations;
    double* offset_ecef_x = oskar_mem_double(x_, &status);
    double* offset_ecef_y = oskar_mem_double(y_, &status);
    double* offset_ecef_z = oskar_mem_double(z_, &status);
    double* ecef_x = (double*)malloc(n*sizeof(double));
    double* ecef_y = (double*)malloc(n*sizeof(double));
    double* ecef_z = (double*)malloc(n*sizeof(double));
    double lon = vis->telescope_lon_deg * M_PI/180.0;
    double lat = vis->telescope_lat_deg * M_PI/180.0;
    /* FIXME altitude hard-coded to 0 as this isn't in the vis structure! */
    double alt = 0;
    double* enu_x = oskar_mem_double(x, &status);
    double* enu_y = oskar_mem_double(y, &status);
    double* enu_z = oskar_mem_double(z, &status);

    oskar_convert_offset_ecef_to_ecef(n, offset_ecef_x, offset_ecef_y,
            offset_ecef_z, lon, lat, alt, ecef_x, ecef_y, ecef_z);
    oskar_convert_ecef_to_enu(n, ecef_x, ecef_y, ecef_z, lon, lat, alt, enu_x,
            enu_y, enu_z);

    free(ecef_x);
    free(ecef_y);
    free(ecef_z);
    oskar_mem_free(x_, &status);
    oskar_mem_free(z_, &status);
    oskar_mem_free(y_, &status);
}

oskar_Mem* oskar_vis_station_horizon_x_metres_create(const oskar_Vis* vis)
{
    int status = OSKAR_SUCCESS;
    oskar_Mem *x = 0, *y = 0, *z = 0;
    int n = vis->num_stations;
    x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    station_xyz_to_horizon(vis, x, y, z);
    oskar_mem_free(y, &status);
    oskar_mem_free(z, &status);
    return x;
}

oskar_Mem* oskar_vis_station_horizon_y_metres_create(const oskar_Vis* vis)
{
    int status = OSKAR_SUCCESS;
    oskar_Mem *x = 0, *y = 0, *z = 0;
    int n = vis->num_stations;
    x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    station_xyz_to_horizon(vis, x, y, z);
    oskar_mem_free(x, &status);
    oskar_mem_free(z, &status);
    return y;
}

oskar_Mem* oskar_vis_station_horizon_z_metres_create(const oskar_Vis* vis)
{
    int status = OSKAR_SUCCESS;
    oskar_Mem *x, *y, *z;
    int n = vis->num_stations;
    x = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    y = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    z = oskar_mem_create(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, (size_t)n, &status);
    station_xyz_to_horizon(vis, x, y, z);
    oskar_mem_free(x, &status);
    oskar_mem_free(y, &status);
    return z;
}


oskar_Mem* oskar_vis_station_lon_deg(oskar_Vis* vis)
{
    return vis->station_lon;
}

const oskar_Mem* oskar_vis_station_lon_deg_const(const oskar_Vis* vis)
{
    return vis->station_lon;
}

oskar_Mem* oskar_vis_station_lat_deg(oskar_Vis* vis)
{
    return vis->station_lat;
}

const oskar_Mem* oskar_vis_station_lat_deg_const(const oskar_Vis* vis)
{
    return vis->station_lat;
}

oskar_Mem* oskar_vis_station_orientation_x_deg(oskar_Vis* vis)
{
    return vis->station_orientation_x;
}

const oskar_Mem* oskar_vis_station_orientation_x_deg_const(const oskar_Vis* vis)
{
    return vis->station_orientation_x;
}

oskar_Mem* oskar_vis_station_orientation_y_deg(oskar_Vis* vis)
{
    return vis->station_orientation_y;
}

const oskar_Mem* oskar_vis_station_orientation_y_deg_const(const oskar_Vis* vis)
{
    return vis->station_orientation_y;
}

oskar_Mem* oskar_vis_baseline_uu_metres(oskar_Vis* vis)
{
    return vis->uu_metres;
}

const oskar_Mem* oskar_vis_baseline_uu_metres_const(const oskar_Vis* vis)
{
    return vis->uu_metres;
}

oskar_Mem* oskar_vis_baseline_vv_metres(oskar_Vis* vis)
{
    return vis->vv_metres;
}

const oskar_Mem* oskar_vis_baseline_vv_metres_const(const oskar_Vis* vis)
{
    return vis->vv_metres;
}

oskar_Mem* oskar_vis_baseline_ww_metres(oskar_Vis* vis)
{
    return vis->ww_metres;
}

const oskar_Mem* oskar_vis_baseline_ww_metres_const(const oskar_Vis* vis)
{
    return vis->ww_metres;
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

void oskar_vis_set_time_inc_seconds(oskar_Vis* vis, double value)
{
    vis->time_inc_seconds = value;
}

void oskar_vis_set_time_int_seconds(oskar_Vis* vis, double value)
{
    vis->time_int_seconds = value;
}

void oskar_vis_set_phase_centre(oskar_Vis* vis, double ra_deg, double dec_deg)
{
    vis->phase_centre_ra_deg = ra_deg;
    vis->phase_centre_dec_deg = dec_deg;
}

void oskar_vis_set_telescope_position(oskar_Vis* vis, double lon_deg,
        double lat_deg)
{
    vis->telescope_lon_deg = lon_deg;
    vis->telescope_lat_deg = lat_deg;
}

#ifdef __cplusplus
}
#endif
