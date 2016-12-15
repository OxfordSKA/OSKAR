/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include "math/oskar_cmath.h"

#include <string.h>

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

double oskar_telescope_polar_motion_x_rad(const oskar_Telescope* model)
{
    return model->pm_x_rad;
}

double oskar_telescope_polar_motion_y_rad(const oskar_Telescope* model)
{
    return model->pm_y_rad;
}

int oskar_telescope_phase_centre_coord_type(const oskar_Telescope* model)
{
    return model->phase_centre_coord_type;
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

int oskar_telescope_pol_mode(const oskar_Telescope* model)
{
    return model->pol_mode;
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

int oskar_telescope_enable_numerical_patterns(const oskar_Telescope* model)
{
    return model->enable_numerical_patterns;
}

int oskar_telescope_max_station_size(const oskar_Telescope* model)
{
    return model->max_station_size;
}

int oskar_telescope_max_station_depth(const oskar_Telescope* model)
{
    return model->max_station_depth;
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

oskar_Mem* oskar_telescope_station_measured_x_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_measured_x_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_measured_x_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_x_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_measured_y_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_measured_y_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_measured_y_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_y_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_measured_z_offset_ecef_metres(
        oskar_Telescope* model)
{
    return model->station_measured_z_offset_ecef_metres;
}

const oskar_Mem* oskar_telescope_station_measured_z_offset_ecef_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_z_offset_ecef_metres;
}

oskar_Mem* oskar_telescope_station_measured_x_enu_metres(oskar_Telescope* model)
{
    return model->station_measured_x_enu_metres;
}

const oskar_Mem* oskar_telescope_station_measured_x_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_x_enu_metres;
}

oskar_Mem* oskar_telescope_station_measured_y_enu_metres(oskar_Telescope* model)
{
    return model->station_measured_y_enu_metres;
}

const oskar_Mem* oskar_telescope_station_measured_y_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_y_enu_metres;
}

oskar_Mem* oskar_telescope_station_measured_z_enu_metres(oskar_Telescope* model)
{
    return model->station_measured_z_enu_metres;
}

const oskar_Mem* oskar_telescope_station_measured_z_enu_metres_const(
        const oskar_Telescope* model)
{
    return model->station_measured_z_enu_metres;
}

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

int oskar_telescope_noise_enabled(const oskar_Telescope* model)
{
    return model->noise_enabled;
}

unsigned int oskar_telescope_noise_seed(const oskar_Telescope* model)
{
    return model->noise_seed;
}


/* Setters. */

void oskar_telescope_set_allow_station_beam_duplication(oskar_Telescope* model,
        int value)
{
    model->allow_station_beam_duplication = value;
}

void oskar_telescope_set_enable_noise(oskar_Telescope* model,
        int value, unsigned int seed)
{
    model->noise_enabled = value;
    model->noise_seed = seed;
}

void oskar_telescope_set_enable_numerical_patterns(oskar_Telescope* model,
        int value)
{
    model->enable_numerical_patterns = value;
}

static void oskar_telescope_set_gaussian_station_beam_p(oskar_Station* station,
        double fwhm_rad, double ref_freq_hz)
{
    oskar_station_set_gaussian_beam_values(station, fwhm_rad, ref_freq_hz);
    if (oskar_station_has_child(station))
    {
        int i, num_elements;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
        {
            oskar_telescope_set_gaussian_station_beam_p(
                    oskar_station_child(station, i), fwhm_rad, ref_freq_hz);
        }
    }
}

void oskar_telescope_set_gaussian_station_beam_width(oskar_Telescope* model,
        double fwhm_deg, double ref_freq_hz)
{
    int i;
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_telescope_set_gaussian_station_beam_p(model->station[i],
                fwhm_deg * M_PI / 180.0, ref_freq_hz);
    }
}

void oskar_telescope_set_noise_freq_file(oskar_Telescope* model,
        const char* filename, int* status)
{
    int i;
    if (*status) return;
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_mem_load_ascii(filename, 1, status,
                oskar_station_noise_freq_hz(model->station[i]), "");
    }
}

void oskar_telescope_set_noise_freq(oskar_Telescope* model,
        double start_hz, double inc_hz, int num_channels, int* status)
{
    int i;
    oskar_Mem* noise_freq_hz;
    noise_freq_hz = oskar_mem_create(model->precision, OSKAR_CPU,
            num_channels, status);
    if (*status) return;
    if (model->precision == OSKAR_DOUBLE)
    {
        double* f = oskar_mem_double(noise_freq_hz, status);
        for (i = 0; i < num_channels; ++i) f[i] = start_hz + i * inc_hz;
    }
    else
    {
        float* f = oskar_mem_float(noise_freq_hz, status);
        for (i = 0; i < num_channels; ++i) f[i] = start_hz + i * inc_hz;
    }

    /* Set noise frequency for all top-level stations. */
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_Mem* t;
        t = oskar_station_noise_freq_hz(model->station[i]);
        oskar_mem_realloc(t, num_channels, status);
        oskar_mem_copy(t, noise_freq_hz, status);
    }
    oskar_mem_free(noise_freq_hz, status);
}

void oskar_telescope_set_noise_rms_file(oskar_Telescope* model,
        const char* filename, int* status)
{
    int i;
    if (*status) return;
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_mem_load_ascii(filename, 1, status,
                oskar_station_noise_rms_jy(model->station[i]), "");
    }
}

void oskar_telescope_set_noise_rms(oskar_Telescope* model,
        double start, double end, int* status)
{
    int i, j, num_channels;
    double inc;
    oskar_Station* s;
    oskar_Mem *noise_rms_jy, *h;

    /* Set noise RMS for all top-level stations. */
    if (*status) return;
    for (i = 0; i < model->num_stations; ++i)
    {
        s = model->station[i];
        h = oskar_station_noise_rms_jy(s);
        num_channels = (int)oskar_mem_length(oskar_station_noise_freq_hz(s));
        if (num_channels == 0)
        {
            *status = OSKAR_ERR_OUT_OF_RANGE;
            return;
        }
        oskar_mem_realloc(h, num_channels, status);
        noise_rms_jy = oskar_mem_create(model->precision, OSKAR_CPU,
                num_channels, status);
        inc = (end - start) / (double)num_channels;
        if (model->precision == OSKAR_DOUBLE)
        {
            double* r = oskar_mem_double(noise_rms_jy, status);
            for (j = 0; j < num_channels; ++j) r[j] = start + j * inc;
        }
        else
        {
            float* r = oskar_mem_float(noise_rms_jy, status);
            for (j = 0; j < num_channels; ++j) r[j] = start + j * inc;
        }
        oskar_mem_copy(h, noise_rms_jy, status);
        oskar_mem_free(noise_rms_jy, status);
    }
}

void oskar_telescope_set_position(oskar_Telescope* model,
        double longitude_rad, double latitude_rad, double altitude_metres)
{
    model->lon_rad = longitude_rad;
    model->lat_rad = latitude_rad;
    model->alt_metres = altitude_metres;
}

void oskar_telescope_set_polar_motion(oskar_Telescope* model,
        double pm_x_rad, double pm_y_rad)
{
    int i;
    model->pm_x_rad = pm_x_rad;
    model->pm_y_rad = pm_y_rad;

    /* Set for all stations, too. */
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_station_set_polar_motion(model->station[i], pm_x_rad, pm_y_rad);
    }
}

static void oskar_telescope_set_station_phase_centre(oskar_Station* station,
        int coord_type, double ra_rad, double dec_rad)
{
    oskar_station_set_phase_centre(station, coord_type, ra_rad, dec_rad);
    if (oskar_station_has_child(station))
    {
        int i, num_elements;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
        {
            oskar_telescope_set_station_phase_centre(
                    oskar_station_child(station, i),
                    coord_type, ra_rad, dec_rad);
        }
    }
}

void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        int coord_type, double ra_rad, double dec_rad)
{
    int i;
    model->phase_centre_coord_type = coord_type;
    model->phase_centre_ra_rad = ra_rad;
    model->phase_centre_dec_rad = dec_rad;
    for (i = 0; i < model->num_stations; ++i)
    {
        oskar_telescope_set_station_phase_centre(model->station[i],
                coord_type, ra_rad, dec_rad);
    }
}

void oskar_telescope_set_channel_bandwidth(oskar_Telescope* model,
        double bandwidth_hz)
{
    model->channel_bandwidth_hz = bandwidth_hz;
}

void oskar_telescope_set_time_average(oskar_Telescope* model,
        double time_average_sec)
{
    model->time_average_sec = time_average_sec;
}

void oskar_telescope_set_station_ids(oskar_Telescope* model)
{
    int i, counter = 0;
    for (i = 0; i < model->num_stations; ++i)
        oskar_station_set_unique_ids(model->station[i], &counter);
}

static void oskar_telescope_set_station_type_p(oskar_Station* station, int type)
{
    oskar_station_set_station_type(station, type);
    if (oskar_station_has_child(station))
    {
        int i, num_elements;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
        {
            oskar_telescope_set_station_type_p(
                    oskar_station_child(station, i), type);
        }
    }
}

void oskar_telescope_set_station_type(oskar_Telescope* model, const char* type,
        int* status)
{
    int i, t;
    if (*status) return;
    if (!strncmp(type, "A", 1) || !strncmp(type, "a", 1))
        t = OSKAR_STATION_TYPE_AA;
    else if (!strncmp(type, "G", 1) || !strncmp(type, "g", 1))
        t = OSKAR_STATION_TYPE_GAUSSIAN_BEAM;
    else if (!strncmp(type, "I", 1) || !strncmp(type, "i", 1))
        t = OSKAR_STATION_TYPE_ISOTROPIC;
    else if (!strncmp(type, "V", 1) || !strncmp(type, "v", 1))
        t = OSKAR_STATION_TYPE_VLA_PBCOR;
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    for (i = 0; i < model->num_stations; ++i)
        oskar_telescope_set_station_type_p(model->station[i], t);
}

void oskar_telescope_set_uv_filter(oskar_Telescope* model,
        double uv_filter_min, double uv_filter_max, const char* units,
        int* status)
{
    if (!strncmp(units, "M", 1) || !strncmp(units, "m", 1))
        model->uv_filter_units = OSKAR_METRES;
    else if (!strncmp(units, "W",  1) || !strncmp(units, "w",  1))
        model->uv_filter_units = OSKAR_WAVELENGTHS;
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    model->uv_filter_min = uv_filter_min;
    model->uv_filter_max = uv_filter_max;
}

void oskar_telescope_set_pol_mode(oskar_Telescope* model, const char* mode,
        int* status)
{
    if (*status) return;
    if (!strncmp(mode, "S", 1) || !strncmp(mode, "s", 1))
        model->pol_mode = OSKAR_POL_MODE_SCALAR;
    else if (!strncmp(mode, "F",  1) || !strncmp(mode, "f",  1))
        model->pol_mode = OSKAR_POL_MODE_FULL;
    else
        *status = OSKAR_ERR_INVALID_ARGUMENT;
}

#ifdef __cplusplus
}
#endif
