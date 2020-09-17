/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "vis/private_vis_header.h"
#include "vis/oskar_vis_header.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Mem* oskar_vis_header_telescope_path(oskar_VisHeader* vis)
{
    return vis->telescope_path;
}

const oskar_Mem* oskar_vis_header_telescope_path_const(const oskar_VisHeader* vis)
{
    return vis->telescope_path;
}

oskar_Mem* oskar_vis_header_settings(oskar_VisHeader* vis)
{
    return vis->settings;
}

const oskar_Mem* oskar_vis_header_settings_const(const oskar_VisHeader* vis)
{
    return vis->settings;
}

int oskar_vis_header_num_tags_per_block(const oskar_VisHeader* vis)
{
    return vis->num_tags_per_block;
}

int oskar_vis_header_write_auto_correlations(const oskar_VisHeader* vis)
{
    return vis->write_autocorr;
}

int oskar_vis_header_write_cross_correlations(const oskar_VisHeader* vis)
{
    return vis->write_crosscorr;
}

int oskar_vis_header_amp_type(const oskar_VisHeader* vis)
{
    return vis->amp_type;
}

int oskar_vis_header_coord_precision(const oskar_VisHeader* vis)
{
    return vis->coord_precision;
}

int oskar_vis_header_max_times_per_block(const oskar_VisHeader* vis)
{
    return vis->max_times_per_block;
}

int oskar_vis_header_max_channels_per_block(const oskar_VisHeader* vis)
{
    return vis->max_channels_per_block;
}

int oskar_vis_header_num_blocks(const oskar_VisHeader* vis)
{
    const int t = (vis->num_times_total + vis->max_times_per_block - 1) /
            vis->max_times_per_block;
    const int c = (vis->num_channels_total + vis->max_channels_per_block - 1) /
            vis->max_channels_per_block;
    return t * c;
}

int oskar_vis_header_num_channels_total(const oskar_VisHeader* vis)
{
    return vis->num_channels_total;
}

int oskar_vis_header_num_times_total(const oskar_VisHeader* vis)
{
    return vis->num_times_total;
}

int oskar_vis_header_num_stations(const oskar_VisHeader* vis)
{
    return vis->num_stations;
}

int oskar_vis_header_pol_type(const oskar_VisHeader* vis)
{
    return vis->pol_type;
}

int oskar_vis_header_phase_centre_coord_type(const oskar_VisHeader* vis)
{
    return vis->phase_centre_type;
}

double oskar_vis_header_phase_centre_ra_deg(const oskar_VisHeader* vis)
{
    return vis->phase_centre_deg[0];
}

double oskar_vis_header_phase_centre_dec_deg(const oskar_VisHeader* vis)
{
    return vis->phase_centre_deg[1];
}

double oskar_vis_header_freq_start_hz(const oskar_VisHeader* vis)
{
    return vis->freq_start_hz;
}

double oskar_vis_header_freq_inc_hz(const oskar_VisHeader* vis)
{
    return vis->freq_inc_hz;
}

double oskar_vis_header_channel_bandwidth_hz(const oskar_VisHeader* vis)
{
    return vis->channel_bandwidth_hz;
}

double oskar_vis_header_time_start_mjd_utc(const oskar_VisHeader* vis)
{
    return vis->time_start_mjd_utc;
}

double oskar_vis_header_time_inc_sec(const oskar_VisHeader* vis)
{
    return vis->time_inc_sec;
}

double oskar_vis_header_time_average_sec(const oskar_VisHeader* vis)
{
    return vis->time_average_sec;
}

double oskar_vis_header_telescope_lon_deg(const oskar_VisHeader* vis)
{
    return vis->telescope_centre_lon_deg;
}

double oskar_vis_header_telescope_lat_deg(const oskar_VisHeader* vis)
{
    return vis->telescope_centre_lat_deg;
}

double oskar_vis_header_telescope_alt_metres(const oskar_VisHeader* vis)
{
    return vis->telescope_centre_alt_m;
}

oskar_Mem* oskar_vis_header_station_x_offset_ecef_metres(oskar_VisHeader* vis)
{
    return vis->station_x_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_header_station_x_offset_ecef_metres_const(
        const oskar_VisHeader* vis)
{
    return vis->station_x_offset_ecef_metres;
}

oskar_Mem* oskar_vis_header_station_y_offset_ecef_metres(oskar_VisHeader* vis)
{
    return vis->station_y_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_header_station_y_offset_ecef_metres_const(
        const oskar_VisHeader* vis)
{
    return vis->station_y_offset_ecef_metres;
}

oskar_Mem* oskar_vis_header_station_z_offset_ecef_metres(oskar_VisHeader* vis)
{
    return vis->station_z_offset_ecef_metres;
}

const oskar_Mem* oskar_vis_header_station_z_offset_ecef_metres_const(
        const oskar_VisHeader* vis)
{
    return vis->station_z_offset_ecef_metres;
}



void oskar_vis_header_set_freq_start_hz(oskar_VisHeader* vis, double value)
{
    vis->freq_start_hz = value;
}

void oskar_vis_header_set_freq_inc_hz(oskar_VisHeader* vis, double value)
{
    vis->freq_inc_hz = value;
}

void oskar_vis_header_set_channel_bandwidth_hz(oskar_VisHeader* vis,
        double value)
{
    vis->channel_bandwidth_hz = value;
}

void oskar_vis_header_set_time_start_mjd_utc(oskar_VisHeader* vis,
        double value)
{
    vis->time_start_mjd_utc = value;
}

void oskar_vis_header_set_time_inc_sec(oskar_VisHeader* vis, double value)
{
    vis->time_inc_sec = value;
}

void oskar_vis_header_set_time_average_sec(oskar_VisHeader* vis, double value)
{
    vis->time_average_sec = value;
}

void oskar_vis_header_set_phase_centre(oskar_VisHeader* vis,
        int coord_type, double ra_deg, double dec_deg)
{
    vis->phase_centre_type = coord_type;
    vis->phase_centre_deg[0] = ra_deg;
    vis->phase_centre_deg[1] = dec_deg;
}

void oskar_vis_header_set_telescope_centre(oskar_VisHeader* vis,
        double lon_deg, double lat_deg, double alt_metres)
{
    vis->telescope_centre_lon_deg = lon_deg;
    vis->telescope_centre_lat_deg = lat_deg;
    vis->telescope_centre_alt_m = alt_metres;
}

void oskar_vis_header_set_pol_type(oskar_VisHeader* vis, int value,
        int* status)
{
    if (oskar_type_is_matrix(vis->amp_type))
    {
        if (value == OSKAR_VIS_POL_TYPE_STOKES_I_Q_U_V ||
                value == OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY)
            vis->pol_type = value;
        else
            *status = OSKAR_ERR_TYPE_MISMATCH;
    }
    else
    {
        if (value == OSKAR_VIS_POL_TYPE_STOKES_I ||
                value == OSKAR_VIS_POL_TYPE_STOKES_Q ||
                value == OSKAR_VIS_POL_TYPE_STOKES_U ||
                value == OSKAR_VIS_POL_TYPE_STOKES_V ||
                value == OSKAR_VIS_POL_TYPE_LINEAR_XX ||
                value == OSKAR_VIS_POL_TYPE_LINEAR_XY ||
                value == OSKAR_VIS_POL_TYPE_LINEAR_YX ||
                value == OSKAR_VIS_POL_TYPE_LINEAR_YY)
            vis->pol_type = value;
        else
            *status = OSKAR_ERR_TYPE_MISMATCH;
    }
}

#ifdef __cplusplus
}
#endif
