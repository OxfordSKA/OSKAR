/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "beam_pattern/oskar_beam_pattern.h"
#include "beam_pattern/private_beam_pattern.h"
#include "beam_pattern/private_beam_pattern_free_device_data.h"
#include "math/oskar_cmath.h"
#include "utility/oskar_device_utils.h"
#include "utility/oskar_get_num_procs.h"

#include <string.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif


int oskar_beam_pattern_num_gpus(const oskar_BeamPattern* h)
{
    return h ? h->num_gpus : 0;
}


void oskar_beam_pattern_set_auto_power_fits(oskar_BeamPattern* h, int flag)
{
    h->auto_power_fits = flag;
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


void oskar_beam_pattern_set_cross_power_raw_text(oskar_BeamPattern* h,
        int flag)
{
    h->cross_power_raw_txt = flag;
}


void oskar_beam_pattern_set_gpus(oskar_BeamPattern* h, int num,
        const int* ids, int* status)
{
    int i, num_gpus_avail;
    if (*status) return;
    oskar_beam_pattern_free_device_data(h, status);
    num_gpus_avail = oskar_device_count(status);
    if (*status) return;
    if (num < 0)
    {
        h->num_gpus = num_gpus_avail;
        h->gpu_ids = (int*) realloc(h->gpu_ids, h->num_gpus * sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->gpu_ids[i] = i;
    }
    else if (num > 0)
    {
        if (num > num_gpus_avail)
        {
            oskar_log_error(h->log, "More GPUs were requested than found.");
            *status = OSKAR_ERR_COMPUTE_DEVICES;
            return;
        }
        h->num_gpus = num;
        h->gpu_ids = (int*) realloc(h->gpu_ids, h->num_gpus * sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->gpu_ids[i] = ids[i];
    }
    else /* num == 0 */
    {
        free(h->gpu_ids);
        h->gpu_ids = 0;
        h->num_gpus = 0;
    }
    for (i = 0; i < h->num_gpus; ++i)
    {
        oskar_device_set(h->gpu_ids[i], status);
        if (*status) return;
    }
}


void oskar_beam_pattern_set_image_size(oskar_BeamPattern* h,
        int width, int height)
{
    h->width = width;
    h->height = height;
}


void oskar_beam_pattern_set_image_fov(oskar_BeamPattern* h,
        double width_deg, double height_deg)
{
    h->fov_deg[0] = width_deg;
    h->fov_deg[1] = height_deg;
}


void oskar_beam_pattern_set_log(oskar_BeamPattern* h, oskar_Log* log)
{
    h->log = log;
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
        value = (h->num_gpus == 0) ? (oskar_get_num_procs() - 1) : h->num_gpus;
    if (value < 1) value = 1;
    h->num_devices = value;
    h->d = (DeviceData*) realloc(h->d, h->num_devices * sizeof(DeviceData));
    memset(h->d, 0, h->num_devices * sizeof(DeviceData));
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
    h->root_path = (char*) realloc(h->root_path, 1 + strlen(path));
    strcpy(h->root_path, path);
}


void oskar_beam_pattern_set_separate_time_and_channel(oskar_BeamPattern* h,
        int flag)
{
    h->separate_time_and_channel = flag;
}


void oskar_beam_pattern_set_sky_model_file(oskar_BeamPattern* h,
        const char* path)
{
    h->sky_model_file = (char*) realloc(h->sky_model_file, 1 + strlen(path));
    strcpy(h->sky_model_file, path);
}


void oskar_beam_pattern_set_station_ids(oskar_BeamPattern* h,
        int num_stations, const int* ids)
{
    int i;
    h->num_active_stations = num_stations;
    if (num_stations < 0) return;
    h->station_ids = (int*) realloc(h->station_ids, num_stations * sizeof(int));
    for (i = 0; i < num_stations; ++i) h->station_ids[i] = ids[i];
}


void oskar_beam_pattern_set_stokes(oskar_BeamPattern* h, const char* stokes)
{
    h->stokes[0] = (strchr(stokes, 'I') || strchr(stokes, 'i'));
    h->stokes[1] = (strchr(stokes, 'Q') || strchr(stokes, 'q'));
    h->stokes[2] = (strchr(stokes, 'U') || strchr(stokes, 'u'));
    h->stokes[3] = (strchr(stokes, 'V') || strchr(stokes, 'v'));
}


void oskar_beam_pattern_set_telescope_model(oskar_BeamPattern* h,
        const oskar_Telescope* model, int* status)
{
    int num_stations;
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
        int i;
        h->num_active_stations = num_stations;
        h->station_ids = (int*)
                realloc(h->station_ids, num_stations * sizeof(int));
        for (i = 0; i < num_stations; ++i) h->station_ids[i] = i;
    }

    /* Remove any existing telescope model, and copy the new one. */
    oskar_telescope_free(h->tel, status);
    h->tel = oskar_telescope_create_copy(model, OSKAR_CPU, status);
    h->pol_mode = oskar_telescope_pol_mode(h->tel);
    h->phase_centre_deg[0] = oskar_telescope_phase_centre_ra_rad(h->tel) *
            180.0 / M_PI;
    h->phase_centre_deg[1] = oskar_telescope_phase_centre_dec_rad(h->tel) *
            180.0 / M_PI;

    /* Analyse the telescope model. */
    oskar_telescope_analyse(h->tel, status);
    if (h->log)
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
