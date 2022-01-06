/*
 * Copyright (c) 2016-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_free_device_data.h"
#include "convert/oskar_convert_cellsize_to_fov.h"
#include "convert/oskar_convert_fov_to_cellsize.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_num_procs.h"

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


oskar_Log* oskar_beam_pattern_log(oskar_BeamPattern* h)
{
    return h->log;
}


int oskar_beam_pattern_num_gpus(const oskar_BeamPattern* h)
{
    return h ? h->num_gpus : 0;
}


void oskar_beam_pattern_set_auto_power_fits(oskar_BeamPattern* h, int flag)
{
    h->auto_power_fits = flag;
}


void oskar_beam_pattern_set_auto_power_phase_fits(oskar_BeamPattern* h,
        int flag)
{
    h->auto_power_phase_fits = flag;
}


void oskar_beam_pattern_set_auto_power_real_fits(oskar_BeamPattern* h,
        int flag)
{
    h->auto_power_real_fits = flag;
}


void oskar_beam_pattern_set_auto_power_imag_fits(oskar_BeamPattern* h,
        int flag)
{
    h->auto_power_imag_fits = flag;
}


void oskar_beam_pattern_set_auto_power_text(oskar_BeamPattern* h, int flag)
{
    h->auto_power_txt = flag;
}


void oskar_beam_pattern_set_average_time_and_channel(oskar_BeamPattern* h,
        int flag)
{
    h->average_time_and_channel = flag;
}


void oskar_beam_pattern_set_average_single_axis(oskar_BeamPattern* h,
        char option)
{
    h->average_single_axis = option;
}


void oskar_beam_pattern_set_coordinate_frame(oskar_BeamPattern* h, char option)
{
    h->coord_frame_type = option;
}


void oskar_beam_pattern_set_coordinate_type(oskar_BeamPattern* h, char option)
{
    h->coord_grid_type = option;
}


void oskar_beam_pattern_set_cross_power_amp_fits(oskar_BeamPattern* h, int flag)
{
    h->cross_power_amp_fits = flag;
}


void oskar_beam_pattern_set_cross_power_amp_text(oskar_BeamPattern* h, int flag)
{
    h->cross_power_amp_txt = flag;
}


void oskar_beam_pattern_set_cross_power_phase_fits(oskar_BeamPattern* h,
        int flag)
{
    h->cross_power_phase_fits = flag;
}


void oskar_beam_pattern_set_cross_power_phase_text(oskar_BeamPattern* h,
        int flag)
{
    h->cross_power_phase_txt = flag;
}


void oskar_beam_pattern_set_cross_power_real_fits(oskar_BeamPattern* h, int flag)
{
    h->cross_power_real_fits = flag;
}


void oskar_beam_pattern_set_cross_power_imag_fits(oskar_BeamPattern* h, int flag)
{
    h->cross_power_imag_fits = flag;
}


void oskar_beam_pattern_set_cross_power_raw_text(oskar_BeamPattern* h,
        int flag)
{
    h->cross_power_raw_txt = flag;
}


void oskar_beam_pattern_set_gpus(oskar_BeamPattern* h, int num,
        const int* ids, int* status)
{
    int i = 0;
    if (*status) return;
    oskar_beam_pattern_free_device_data(h, status);
    if (*status) return;
    if (num < 0)
    {
        free(h->gpu_ids);
        h->gpu_ids = (int*) calloc(h->num_gpus_avail, sizeof(int));
        h->num_gpus = 0;
        if (h->gpu_ids)
        {
            h->num_gpus = h->num_gpus_avail;
            for (i = 0; i < h->num_gpus; ++i) h->gpu_ids[i] = i;
        }
    }
    else if (num > 0)
    {
        if (num > h->num_gpus_avail)
        {
            oskar_log_error(h->log, "More GPUs were requested than found.");
            *status = OSKAR_ERR_COMPUTE_DEVICES;
            return;
        }
        free(h->gpu_ids);
        h->gpu_ids = (int*) calloc(num, sizeof(int));
        h->num_gpus = 0;
        if (h->gpu_ids)
        {
            h->num_gpus = num;
            for (i = 0; i < h->num_gpus; ++i) h->gpu_ids[i] = ids[i];
        }
    }
    else /* num == 0 */
    {
        free(h->gpu_ids);
        h->gpu_ids = 0;
        h->num_gpus = 0;
    }
    for (i = 0; (i < h->num_gpus) && h->gpu_ids; ++i)
    {
        oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        if (*status) return;
    }
}


void oskar_beam_pattern_set_image_size(oskar_BeamPattern* h,
        int width, int height)
{
    h->width = width;
    h->height = height;
    if (h->set_cellsize)
    {
        h->fov_deg[0] = oskar_convert_cellsize_to_fov(
                h->cellsize_rad, h->width) * (180.0 / M_PI);
        h->fov_deg[1] = oskar_convert_cellsize_to_fov(
                h->cellsize_rad, h->height) * (180.0 / M_PI);
    }
    else
    {
        h->cellsize_rad = oskar_convert_fov_to_cellsize(
                h->fov_deg[0] * (M_PI / 180.0), h->width);
    }
}


void oskar_beam_pattern_set_image_cellsize(oskar_BeamPattern* h,
        double cellsize_arcsec)
{
    h->set_cellsize = 1;
    h->cellsize_rad = (cellsize_arcsec / 3600.0) * (M_PI / 180.0);
    h->fov_deg[0] = oskar_convert_cellsize_to_fov(
            h->cellsize_rad, h->width) * (180.0 / M_PI);
    h->fov_deg[1] = oskar_convert_cellsize_to_fov(
            h->cellsize_rad, h->height) * (180.0 / M_PI);
}


void oskar_beam_pattern_set_image_fov(oskar_BeamPattern* h,
        double width_deg, double height_deg)
{
    h->set_cellsize = 0;
    h->fov_deg[0] = width_deg;
    h->fov_deg[1] = height_deg;
    h->cellsize_rad = oskar_convert_fov_to_cellsize(
            h->fov_deg[0] * (M_PI / 180.0), h->width);
}


void oskar_beam_pattern_set_max_chunk_size(oskar_BeamPattern* h, int value)
{
    h->max_chunk_size = value;
}


void oskar_beam_pattern_set_num_devices(oskar_BeamPattern* h, int value)
{
    int status = 0;
    oskar_beam_pattern_free_device_data(h, &status);
    if (value < 1)
    {
        value = (h->num_gpus == 0) ? (oskar_get_num_procs() - 1) : h->num_gpus;
    }
    if (value < 1) value = 1;
    h->num_devices = value;
    free(h->d);
    h->d = (DeviceData*) calloc(h->num_devices, sizeof(DeviceData));
}


void oskar_beam_pattern_set_observation_frequency(oskar_BeamPattern* h,
        double start_hz, double inc_hz, int num_channels)
{
    h->freq_start_hz = start_hz;
    h->freq_inc_hz = inc_hz;
    h->num_channels = num_channels;
}


void oskar_beam_pattern_set_observation_time(oskar_BeamPattern* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps)
{
    h->time_start_mjd_utc = time_start_mjd_utc;
    h->time_inc_sec = inc_sec;
    h->num_time_steps = num_time_steps;
}


void oskar_beam_pattern_set_root_path(oskar_BeamPattern* h, const char* path)
{
    if (!path) return;
    const size_t buffer_size = 1 + strlen(path);
    free(h->root_path);
    h->root_path = (char*) calloc(buffer_size, sizeof(char));
    if (h->root_path) memcpy(h->root_path, path, buffer_size);
}


void oskar_beam_pattern_set_separate_time_and_channel(oskar_BeamPattern* h,
        int flag)
{
    h->separate_time_and_channel = flag;
}


void oskar_beam_pattern_set_sky_model_file(oskar_BeamPattern* h,
        const char* path)
{
    if (!path) return;
    const size_t buffer_size = 1 + strlen(path);
    free(h->sky_model_file);
    h->sky_model_file = (char*) calloc(buffer_size, sizeof(char));
    if (h->sky_model_file) memcpy(h->sky_model_file, path, buffer_size);
}


void oskar_beam_pattern_set_station_ids(oskar_BeamPattern* h,
        int num_stations, const int* ids)
{
    int i = 0;
    h->num_active_stations = num_stations;
    if (num_stations < 0) return;
    free(h->station_ids);
    h->station_ids = (int*) calloc(num_stations, sizeof(int));
    if (h->station_ids)
    {
        for (i = 0; i < num_stations; ++i) h->station_ids[i] = ids[i];
    }
}


void oskar_beam_pattern_set_test_source_stokes_i(oskar_BeamPattern* h,
        int enabled)
{
    h->stokes[0] = enabled;
}


void oskar_beam_pattern_set_test_source_stokes_custom(oskar_BeamPattern* h,
        int enabled, double i, double q, double u, double v, int* status)
{
    if (*status) return;
    if (i < 0.0 || ((q*q + u*u + v*v) > i*i))
    {
        oskar_log_error(h->log, "Invalid Stokes parameters for test source "
                "(I, Q, U, V) = (%.2e, %.2e, %.2e, %.2e)", i, q, u, v);
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }
    h->stokes[1] = enabled;
    h->test_source_stokes[0] = i;
    h->test_source_stokes[1] = q;
    h->test_source_stokes[2] = u;
    h->test_source_stokes[3] = v;
}


void oskar_beam_pattern_set_telescope_model(oskar_BeamPattern* h,
        const oskar_Telescope* model, int* status)
{
    int num_stations = 0;
    if (*status || !h || !model) return;

    /* Check the model is not empty. */
    num_stations = oskar_telescope_num_stations(model);
    if (num_stations == 0)
    {
        oskar_log_error(h->log, "Telescope model is empty.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }
    if (h->num_active_stations < 0)
    {
        int i = 0;
        h->num_active_stations = num_stations;
        h->station_ids = (int*)
                realloc(h->station_ids, num_stations * sizeof(int));
        for (i = 0; i < num_stations; ++i) h->station_ids[i] = i;
    }

    /* Remove any existing telescope model, and copy the new one. */
    oskar_telescope_free(h->tel, status);
    h->tel = oskar_telescope_create_copy(model, OSKAR_CPU, status);
    h->pol_mode = oskar_telescope_pol_mode(h->tel);
    h->phase_centre_deg[0] = oskar_telescope_phase_centre_longitude_rad(h->tel)
            * 180.0 / M_PI;
    h->phase_centre_deg[1] = oskar_telescope_phase_centre_latitude_rad(h->tel)
            * 180.0 / M_PI;

    /* Analyse the telescope model. */
    oskar_telescope_analyse(h->tel, status);
    oskar_telescope_log_summary(h->tel, h->log, status);
}


void oskar_beam_pattern_set_voltage_amp_fits(oskar_BeamPattern* h, int flag)
{
    h->voltage_amp_fits = flag;
}


void oskar_beam_pattern_set_voltage_amp_text(oskar_BeamPattern* h, int flag)
{
    h->voltage_amp_txt = flag;
}


void oskar_beam_pattern_set_voltage_phase_fits(oskar_BeamPattern* h, int flag)
{
    h->voltage_phase_fits = flag;
}


void oskar_beam_pattern_set_voltage_phase_text(oskar_BeamPattern* h, int flag)
{
    h->voltage_phase_txt = flag;
}


void oskar_beam_pattern_set_voltage_raw_text(oskar_BeamPattern* h, int flag)
{
    h->voltage_raw_txt = flag;
}

#ifdef __cplusplus
}
#endif
