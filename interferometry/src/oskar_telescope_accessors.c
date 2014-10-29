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

#include <private_telescope.h>

#include <oskar_telescope_accessors.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Properties and metadata. */

int oskar_telescope_precision(const oskar_Telescope* model)
{
    return model->precision;
}

int oskar_telescope_mem_location(const oskar_Telescope* model)
{
    return model->mem_location;
}

double oskar_telescope_lon_rad(const oskar_Telescope* model)
{
    return model->lon_rad;
}

double oskar_telescope_lat_rad(const oskar_Telescope* model)
{
    return model->lat_rad;
}

double oskar_telescope_alt_metres(const oskar_Telescope* model)
{
    return model->alt_metres;
}

double oskar_telescope_phase_centre_ra_rad(const oskar_Telescope* model)
{
    return model->phase_centre_ra_rad;
}

double oskar_telescope_phase_centre_dec_rad(const oskar_Telescope* model)
{
    return model->phase_centre_dec_rad;
}

double oskar_telescope_channel_bandwidth_hz(const oskar_Telescope* model)
{
    return model->channel_bandwidth_hz;
}

double oskar_telescope_time_average_sec(const oskar_Telescope* model)
{
    return model->time_average_sec;
}

double oskar_telescope_uv_filter_min(const oskar_Telescope* model)
{
    return model->uv_filter_min;
}

double oskar_telescope_uv_filter_max(const oskar_Telescope* model)
{
    return model->uv_filter_max;
}

int oskar_telescope_uv_filter_units(const oskar_Telescope* model)
{
    return model->uv_filter_units;
}

int oskar_telescope_num_baselines(const oskar_Telescope* model)
{
    return ((model->num_stations * (model->num_stations - 1)) / 2);
}

int oskar_telescope_num_stations(const oskar_Telescope* model)
{
    return model->num_stations;
}

int oskar_telescope_identical_stations(const oskar_Telescope* model)
{
    return model->identical_stations;
}

int oskar_telescope_allow_station_beam_duplication(
        const oskar_Telescope* model)
{
    return model->allow_station_beam_duplication;
}

int oskar_telescope_max_station_size(const oskar_Telescope* model)
{
    return model->max_station_size;
}

int oskar_telescope_max_station_depth(const oskar_Telescope* model)
{
    return model->max_station_depth;
}

int oskar_telescope_random_seed(const oskar_Telescope* model)
{
    return model->seed_time_variable_station_element_errors;
}


/* Station models. */

oskar_Station* oskar_telescope_station(oskar_Telescope* model, int i)
{
    return model->station[i];
}

const oskar_Station* oskar_telescope_station_const(
        const oskar_Telescope* model, int i)
{
    return model->station[i];
}


/* Coordinate arrays. */

oskar_Mem* oskar_telescope_station_true_x_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_true_x_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_true_x_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_x_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_true_y_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_true_y_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_true_y_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_y_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_true_z_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_true_z_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_true_z_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_z_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_true_x_enu_metres(oskar_Telescope* model)
{
    return model->station_true_x_enu_metres;
}

const oskar_Mem* oskar_telescope_station_true_x_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_x_enu_metres;
}

oskar_Mem* oskar_telescope_station_true_y_enu_metres(oskar_Telescope* model)
{
    return model->station_true_y_enu_metres;
}

const oskar_Mem* oskar_telescope_station_true_y_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_y_enu_metres;
}

oskar_Mem* oskar_telescope_station_true_z_enu_metres(oskar_Telescope* model)
{
    return model->station_true_z_enu_metres;
}

const oskar_Mem* oskar_telescope_station_true_z_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_true_z_enu_metres;
}


/* Setters. */

void oskar_telescope_set_position(oskar_Telescope* model,
        double longitude_rad, double latitude_rad, double altitude_metres)
{
    model->lon_rad = longitude_rad;
    model->lat_rad = latitude_rad;
    model->alt_metres = altitude_metres;
}

void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        double ra_rad, double dec_rad)
{
    model->phase_centre_ra_rad = ra_rad;
    model->phase_centre_dec_rad = dec_rad;
}

void oskar_telescope_set_smearing_values(oskar_Telescope* model,
        double bandwidth_hz, double time_average_sec)
{
    model->channel_bandwidth_hz = bandwidth_hz;
    model->time_average_sec = time_average_sec;
}

void oskar_telescope_set_uv_filter(oskar_Telescope* model,
        double uv_filter_min, double uv_filter_max, int uv_filter_units)
{
    model->uv_filter_min = uv_filter_min;
    model->uv_filter_max = uv_filter_max;
    model->uv_filter_units = uv_filter_units;
}

void oskar_telescope_set_allow_station_beam_duplication(oskar_Telescope* model, int value)
{
    model->allow_station_beam_duplication = value;
}

void oskar_telescope_set_random_seed(oskar_Telescope* model, int value)
{
    model->seed_time_variable_station_element_errors = value;
}


#ifdef __cplusplus
}
#endif
