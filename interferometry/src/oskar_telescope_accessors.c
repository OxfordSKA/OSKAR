/*
 * Copyright (c) 2013, The University of Oxford
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

/* Refactor to precision. */
int oskar_telescope_type(const oskar_Telescope* model)
{
    return model->precision;
}

int oskar_telescope_location(const oskar_Telescope* model)
{
    return model->location;
}

double oskar_telescope_longitude_rad(const oskar_Telescope* model)
{
    return model->longitude_rad;
}

double oskar_telescope_latitude_rad(const oskar_Telescope* model)
{
    return model->latitude_rad;
}

double oskar_telescope_altitude_m(const oskar_Telescope* model)
{
    return model->altitude_m;
}

double oskar_telescope_ra0_rad(const oskar_Telescope* model)
{
    return model->ra0_rad;
}

double oskar_telescope_dec0_rad(const oskar_Telescope* model)
{
    return model->dec0_rad;
}

double oskar_telescope_bandwidth_hz(const oskar_Telescope* model)
{
    return model->bandwidth_hz;
}

double oskar_telescope_time_average_sec(
        const oskar_Telescope* model)
{
    return model->time_average_sec;
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

int oskar_telescope_common_horizon(const oskar_Telescope* model)
{
    return model->use_common_sky;
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

oskar_Mem* oskar_telescope_station_x(oskar_Telescope* model)
{
    return &model->station_x;
}

const oskar_Mem* oskar_telescope_station_x_const(const oskar_Telescope* model)
{
    return &model->station_x;
}

oskar_Mem* oskar_telescope_station_y(oskar_Telescope* model)
{
    return &model->station_y;
}

const oskar_Mem* oskar_telescope_station_y_const(const oskar_Telescope* model)
{
    return &model->station_y;
}

oskar_Mem* oskar_telescope_station_z(oskar_Telescope* model)
{
    return &model->station_z;
}

const oskar_Mem* oskar_telescope_station_z_const(const oskar_Telescope* model)
{
    return &model->station_z;
}

oskar_Mem* oskar_telescope_station_x_hor(oskar_Telescope* model)
{
    return &model->station_x_hor;
}

const oskar_Mem* oskar_telescope_station_x_hor_const(
        const oskar_Telescope* model)
{
    return &model->station_x_hor;
}

oskar_Mem* oskar_telescope_station_y_hor(oskar_Telescope* model)
{
    return &model->station_y_hor;
}

const oskar_Mem* oskar_telescope_station_y_hor_const(
        const oskar_Telescope* model)
{
    return &model->station_y_hor;
}

oskar_Mem* oskar_telescope_station_z_hor(oskar_Telescope* model)
{
    return &model->station_z_hor;
}

const oskar_Mem* oskar_telescope_station_z_hor_const(
        const oskar_Telescope* model)
{
    return &model->station_z_hor;
}


/* Setters. */

void oskar_telescope_set_position(oskar_Telescope* model,
        double longitude_rad, double latitude_rad, double altitude_m)
{
    model->longitude_rad = longitude_rad;
    model->latitude_rad = latitude_rad;
    model->altitude_m = altitude_m;
}

void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        double ra_rad, double dec_rad)
{
    model->ra0_rad = ra_rad;
    model->dec0_rad = dec_rad;
}

void oskar_telescope_set_smearing_values(oskar_Telescope* model,
        double bandwidth_hz, double time_average_sec)
{
    model->bandwidth_hz = bandwidth_hz;
    model->time_average_sec = time_average_sec;
}

void oskar_telescope_set_common_horizon(oskar_Telescope* model, int value)
{
    model->use_common_sky = value;
}

void oskar_telescope_set_random_seed(oskar_Telescope* model, int value)
{
    model->seed_time_variable_station_element_errors = value;
}


#ifdef __cplusplus
}
#endif
