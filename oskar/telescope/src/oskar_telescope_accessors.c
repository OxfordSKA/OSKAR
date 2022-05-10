/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_find_closest_match.h"
#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include "math/oskar_cmath.h"

#include <ctype.h>
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

double oskar_telescope_phase_centre_longitude_rad(const oskar_Telescope* model)
{
    return model->phase_centre_rad[0];
}

double oskar_telescope_phase_centre_latitude_rad(const oskar_Telescope* model)
{
    return model->phase_centre_rad[1];
}

double oskar_telescope_channel_bandwidth_hz(const oskar_Telescope* model)
{
    return model->channel_bandwidth_hz;
}

double oskar_telescope_tec_screen_height_km(const oskar_Telescope* model)
{
    return model->tec_screen_height_km;
}

double oskar_telescope_tec_screen_pixel_size_m(const oskar_Telescope* model)
{
    return model->tec_screen_pixel_size_m;
}

double oskar_telescope_tec_screen_time_interval_sec(
        const oskar_Telescope* model)
{
    return model->tec_screen_time_interval_sec;
}

int oskar_telescope_isoplanatic_screen(const oskar_Telescope* model)
{
    return model->isoplanatic_screen;
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

int oskar_telescope_num_station_models(const oskar_Telescope* model)
{
    return model->num_station_models;
}

int oskar_telescope_allow_station_beam_duplication(
        const oskar_Telescope* model)
{
    return model->allow_station_beam_duplication;
}

char oskar_telescope_ionosphere_screen_type(const oskar_Telescope* model)
{
    return (char) (model->ionosphere_screen_type);
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
    if (i >= model->num_station_models) return 0;
    return model->station[i];
}

const oskar_Station* oskar_telescope_station_const(
        const oskar_Telescope* model, int i)
{
    if (i >= model->num_station_models) return 0;
    return model->station[i];
}


/* Arrays. */

oskar_Mem* oskar_telescope_station_type_map(oskar_Telescope* model)
{
    return model->station_type_map;
}

const oskar_Mem* oskar_telescope_station_type_map_const(
        const oskar_Telescope* model)
{
    return model->station_type_map;
}

oskar_Mem* oskar_telescope_station_measured_offset_ecef_metres(
        oskar_Telescope* model, int dim)
{
    return model->station_measured_offset_ecef_metres[dim];
}

const oskar_Mem* oskar_telescope_station_measured_offset_ecef_metres_const(
        const oskar_Telescope* model, int dim)
{
    return model->station_measured_offset_ecef_metres[dim];
}

oskar_Mem* oskar_telescope_station_measured_enu_metres(
        oskar_Telescope* model, int dim)
{
    return model->station_measured_enu_metres[dim];
}

const oskar_Mem* oskar_telescope_station_measured_enu_metres_const(
        const oskar_Telescope* model, int dim)
{
    return model->station_measured_enu_metres[dim];
}

oskar_Mem* oskar_telescope_station_true_offset_ecef_metres(
        oskar_Telescope* model, int dim)
{
    return model->station_true_offset_ecef_metres[dim];
}

const oskar_Mem* oskar_telescope_station_true_offset_ecef_metres_const(
        const oskar_Telescope* model, int dim)
{
    return model->station_true_offset_ecef_metres[dim];
}

oskar_Mem* oskar_telescope_station_true_enu_metres(
        oskar_Telescope* model, int dim)
{
    return model->station_true_enu_metres[dim];
}

const oskar_Mem* oskar_telescope_station_true_enu_metres_const(
        const oskar_Telescope* model, int dim)
{
    return model->station_true_enu_metres[dim];
}

int oskar_telescope_noise_enabled(const oskar_Telescope* model)
{
    return model->noise_enabled;
}

unsigned int oskar_telescope_noise_seed(const oskar_Telescope* model)
{
    return model->noise_seed;
}

const char* oskar_telescope_tec_screen_path(const oskar_Telescope* model)
{
    return oskar_mem_char_const(model->tec_screen_path);
}

oskar_Gains* oskar_telescope_gains(oskar_Telescope* model)
{
    return model->gains;
}

const oskar_Gains* oskar_telescope_gains_const(const oskar_Telescope* model)
{
    return model->gains;
}

oskar_Harp* oskar_telescope_harp_data(oskar_Telescope* model,
        double freq_hz)
{
    int index = 0, status = 0;
    if (!model || !model->harp_data) return 0;
    index = oskar_find_closest_match(freq_hz, model->harp_freq_cpu, &status);
    return model->harp_data[index];
}

const oskar_Harp* oskar_telescope_harp_data_const(const oskar_Telescope* model,
        double freq_hz)
{
    int index = 0, status = 0;
    if (!model || !model->harp_data) return 0;
    index = oskar_find_closest_match(freq_hz, model->harp_freq_cpu, &status);
    return model->harp_data[index];
}


/* Setters. */

void oskar_telescope_set_allow_station_beam_duplication(oskar_Telescope* model,
        int value)
{
    model->allow_station_beam_duplication = value;
}

void oskar_telescope_set_ionosphere_screen_type(oskar_Telescope* model,
        const char* type)
{
    model->ionosphere_screen_type = toupper(type[0]);
}

void oskar_telescope_set_isoplanatic_screen(oskar_Telescope* model, int flag)
{
    model->isoplanatic_screen = flag;
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
        int i = 0;
        const int num_elements = oskar_station_num_elements(station);
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
    int i = 0;
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_telescope_set_gaussian_station_beam_p(model->station[i],
                fwhm_deg * M_PI / 180.0, ref_freq_hz);
    }
}

void oskar_telescope_set_noise_freq_file(oskar_Telescope* model,
        const char* filename, int* status)
{
    int i = 0;
    if (*status) return;
    for (i = 0; i < model->num_station_models; ++i)
    {
        if (!model->station[i]) continue;
        oskar_mem_load_ascii(filename, 1, status,
                oskar_station_noise_freq_hz(model->station[i]), "");
    }
}

void oskar_telescope_set_noise_freq(oskar_Telescope* model,
        double start_hz, double inc_hz, int num_channels, int* status)
{
    int i = 0;
    oskar_Mem* noise_freq_hz = 0;
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
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_Mem* t = 0;
        if (!model->station[i]) continue;
        t = oskar_station_noise_freq_hz(model->station[i]);
        oskar_mem_realloc(t, num_channels, status);
        oskar_mem_copy(t, noise_freq_hz, status);
    }
    oskar_mem_free(noise_freq_hz, status);
}

void oskar_telescope_set_noise_rms_file(oskar_Telescope* model,
        const char* filename, int* status)
{
    int i = 0;
    if (*status) return;
    for (i = 0; i < model->num_station_models; ++i)
    {
        if (!model->station[i]) continue;
        oskar_mem_load_ascii(filename, 1, status,
                oskar_station_noise_rms_jy(model->station[i]), "");
    }
}

void oskar_telescope_set_noise_rms(oskar_Telescope* model,
        double start, double end, int* status)
{
    int i = 0, j = 0, num_channels = 0;
    double inc = 0.0;
    oskar_Station* s = 0;
    oskar_Mem *noise_rms_jy = 0, *h = 0;

    /* Set noise RMS for all top-level stations. */
    if (*status) return;
    for (i = 0; i < model->num_station_models; ++i)
    {
        s = model->station[i];
        if (!s) continue;
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
    int i = 0;
    model->pm_x_rad = pm_x_rad;
    model->pm_y_rad = pm_y_rad;

    /* Set for all stations, too. */
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_station_set_polar_motion(model->station[i], pm_x_rad, pm_y_rad);
    }
}

static void oskar_telescope_set_station_phase_centre(oskar_Station* station,
        int coord_type, double longitude_rad, double latitude_rad)
{
    oskar_station_set_phase_centre(station, coord_type, longitude_rad,
            latitude_rad);
    if (oskar_station_has_child(station))
    {
        int i = 0, num_elements = 0;
        num_elements = oskar_station_num_elements(station);
        for (i = 0; i < num_elements; ++i)
        {
            oskar_telescope_set_station_phase_centre(
                    oskar_station_child(station, i),
                    coord_type, longitude_rad, latitude_rad);
        }
    }
}

void oskar_telescope_set_phase_centre(oskar_Telescope* model,
        int coord_type, double longitude_rad, double latitude_rad)
{
    int i = 0;
    model->phase_centre_coord_type = coord_type;
    model->phase_centre_rad[0] = longitude_rad;
    model->phase_centre_rad[1] = latitude_rad;
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_telescope_set_station_phase_centre(model->station[i],
                coord_type, longitude_rad, latitude_rad);
    }
}

void oskar_telescope_set_tec_screen_height(oskar_Telescope* model,
        double height_km)
{
    model->tec_screen_height_km = height_km;
}

void oskar_telescope_set_tec_screen_pixel_size(oskar_Telescope* model,
        double pixel_size_m)
{
    model->tec_screen_pixel_size_m = pixel_size_m;
}

void oskar_telescope_set_tec_screen_time_interval(oskar_Telescope* model,
        double time_interval_sec)
{
    model->tec_screen_time_interval_sec = time_interval_sec;
}

void oskar_telescope_set_tec_screen_path(oskar_Telescope* model,
        const char* path)
{
    int status = 0;
    const size_t len = 1 + strlen(path);
    oskar_mem_realloc(model->tec_screen_path, len, &status);
    memcpy(oskar_mem_void(model->tec_screen_path), path, len);
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

static void oskar_telescope_set_station_type_p(oskar_Station* station, int type)
{
    oskar_station_set_station_type(station, type);
    if (oskar_station_has_child(station))
    {
        int i = 0, num_elements = 0;
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
    int i = 0, t = 0;
    if (*status) return;
    if (!strncmp(type, "A", 1) || !strncmp(type, "a", 1))
    {
        t = OSKAR_STATION_TYPE_AA;
    }
    else if (!strncmp(type, "G", 1) || !strncmp(type, "g", 1))
    {
        t = OSKAR_STATION_TYPE_GAUSSIAN_BEAM;
    }
    else if (!strncmp(type, "I", 1) || !strncmp(type, "i", 1))
    {
        t = OSKAR_STATION_TYPE_ISOTROPIC;
    }
    else if (!strncmp(type, "V", 1) || !strncmp(type, "v", 1))
    {
        t = OSKAR_STATION_TYPE_VLA_PBCOR;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    for (i = 0; i < model->num_station_models; ++i)
    {
        oskar_telescope_set_station_type_p(model->station[i], t);
    }
}

void oskar_telescope_set_unique_stations(oskar_Telescope* model,
        int value, int* status)
{
    if (value)
    {
        int i = 0;
        int* type_map = oskar_mem_int(model->station_type_map, status);
        for (i = 0; i < model->num_stations; ++i) type_map[i] = i;
    }
    else
    {
        oskar_mem_clear_contents(model->station_type_map, status);
    }
}

void oskar_telescope_set_uv_filter(oskar_Telescope* model,
        double uv_filter_min, double uv_filter_max, const char* units,
        int* status)
{
    if (!strncmp(units, "M", 1) || !strncmp(units, "m", 1))
    {
        model->uv_filter_units = OSKAR_METRES;
    }
    else if (!strncmp(units, "W",  1) || !strncmp(units, "w",  1))
    {
        model->uv_filter_units = OSKAR_WAVELENGTHS;
    }
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
    {
        model->pol_mode = OSKAR_POL_MODE_SCALAR;
    }
    else if (!strncmp(mode, "F",  1) || !strncmp(mode, "f",  1))
    {
        model->pol_mode = OSKAR_POL_MODE_FULL;
    }
    else
    {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
    }
}

#ifdef __cplusplus
}
#endif
