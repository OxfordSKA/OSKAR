/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "interferometer/private_interferometer.h"
#include "interferometer/oskar_interferometer.h"
#include "utility/oskar_get_num_procs.h"
#include "utility/oskar_device.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_interferometer_coords_only(const oskar_Interferometer* h)
{
    return h->coords_only;
}

oskar_Log* oskar_interferometer_log(oskar_Interferometer* h)
{
    return h->log;
}

int oskar_interferometer_num_devices(const oskar_Interferometer* h)
{
    return h ? h->num_devices : 0;
}

int oskar_interferometer_num_gpus(const oskar_Interferometer* h)
{
    return h ? h->num_gpus : 0;
}

int oskar_interferometer_num_vis_blocks(const oskar_Interferometer* h)
{
    if (h->num_time_steps == 0 || h->num_channels == 0)
        return 0;
    const int t = (h->num_time_steps + h->max_times_per_block - 1) /
            h->max_times_per_block;
    const int c = (h->num_channels + h->max_channels_per_block - 1) /
            h->max_channels_per_block;
    return t * c;
}

void oskar_interferometer_reset_work_unit_index(oskar_Interferometer* h)
{
    h->work_unit_index = 0;
}

void oskar_interferometer_set_coords_only(oskar_Interferometer* h, int value,
        int* status)
{
    (void)status;
    h->coords_only = value;
}

void oskar_interferometer_set_correlation_type(oskar_Interferometer* h,
        const char* type, int* status)
{
    if (*status) return;
    if (!strncmp(type, "A", 1) || !strncmp(type, "a", 1))
        h->correlation_type = 'A';
    else if (!strncmp(type, "B",  1) || !strncmp(type, "b",  1))
        h->correlation_type = 'B';
    else if (!strncmp(type, "C",  1) || !strncmp(type, "c",  1))
        h->correlation_type = 'C';
    else *status = OSKAR_ERR_INVALID_ARGUMENT;
}

void oskar_interferometer_set_force_polarised_ms(oskar_Interferometer* h,
        int value)
{
    h->force_polarised_ms = value;
}

void oskar_interferometer_set_gpus(oskar_Interferometer* h, int num,
        const int* ids, int* status)
{
    int i;
    if (*status || !h) return;
    oskar_interferometer_free_device_data(h, status);
    if (*status) return;
    if (num < 0)
    {
        h->num_gpus = h->num_gpus_avail;
        h->gpu_ids = (int*) realloc(h->gpu_ids, h->num_gpus * sizeof(int));
        for (i = 0; i < h->num_gpus; ++i)
            h->gpu_ids[i] = i;
    }
    else if (num > 0)
    {
        if (num > h->num_gpus_avail)
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
        oskar_device_set(h->dev_loc, h->gpu_ids[i], status);
        if (*status) return;
    }
}

void oskar_interferometer_set_horizon_clip(oskar_Interferometer* h, int value)
{
    h->apply_horizon_clip = value;
}

void oskar_interferometer_set_ignore_w_components(oskar_Interferometer* h,
        int value)
{
    h->ignore_w_components = value;
}

void oskar_interferometer_set_max_sources_per_chunk(oskar_Interferometer* h,
        int value)
{
    h->max_sources_per_chunk = value;
}

void oskar_interferometer_set_max_channels_per_block(oskar_Interferometer* h,
        int value)
{
    h->max_channels_per_block = value;
}

void oskar_interferometer_set_max_times_per_block(oskar_Interferometer* h,
        int value)
{
    h->max_times_per_block = value;
}

void oskar_interferometer_set_num_devices(oskar_Interferometer* h, int value)
{
    int status = 0;
    oskar_interferometer_free_device_data(h, &status);
    if (value < 1)
        value = (h->num_gpus == 0) ? oskar_get_num_procs() : h->num_gpus;
    if (value < 1) value = 1;
    h->num_devices = value;
    h->d = (DeviceData*) realloc(h->d, h->num_devices * sizeof(DeviceData));
    memset(h->d, 0, h->num_devices * sizeof(DeviceData));
}

void oskar_interferometer_set_observation_frequency(oskar_Interferometer* h,
        double start_hz, double inc_hz, int num_channels)
{
    h->freq_start_hz = start_hz;
    h->freq_inc_hz = inc_hz;
    h->num_channels = num_channels;
    if (h->max_channels_per_block <= 0)
        h->max_channels_per_block = (num_channels < 8) ? num_channels : 8;
}

void oskar_interferometer_set_observation_time(oskar_Interferometer* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps)
{
    h->time_start_mjd_utc = time_start_mjd_utc;
    h->time_inc_sec = inc_sec;
    h->num_time_steps = num_time_steps;
    if (h->max_times_per_block <= 0)
        h->max_times_per_block = (num_time_steps < 8) ? num_time_steps : 8;
}

void oskar_interferometer_set_settings_path(oskar_Interferometer* h,
        const char* filename)
{
    if (!filename) return;
    const int len = (int) strlen(filename);
    if (len == 0) return;
    free(h->settings_path);
    h->settings_path = (char*) calloc(1 + len, 1);
    strcpy(h->settings_path, filename);
}

void oskar_interferometer_set_sky_model(oskar_Interferometer* h,
        const oskar_Sky* sky, int* status)
{
    int i;
    if (*status || !h || !sky) return;

    /* Clear the old chunk set. */
    for (i = 0; i < h->num_sky_chunks; ++i)
        oskar_sky_free(h->sky_chunks[i], status);
    free(h->sky_chunks);
    h->sky_chunks = 0;
    h->num_sky_chunks = 0;

    /* Split up the sky model into chunks and store them. */
    h->num_sources_total = oskar_sky_num_sources(sky);
    if (h->num_sources_total > 0)
        oskar_sky_append_to_set(&h->num_sky_chunks, &h->sky_chunks,
                h->max_sources_per_chunk, sky, status);
    h->init_sky = 0;

    /* Print summary data. */
    oskar_log_section(h->log, 'M', "Sky model summary");
    oskar_log_value(h->log, 'M', 0, "Num. sources", "%d", h->num_sources_total);
    oskar_log_value(h->log, 'M', 0, "Num. chunks", "%d", h->num_sky_chunks);
    if (h->num_sources_total < 32 && h->num_gpus > 0)
        oskar_log_advice(h->log, "It may be faster to use CPU cores "
                "only, as the sky model contains fewer than 32 sources.");
}

void oskar_interferometer_set_telescope_model(oskar_Interferometer* h,
        const oskar_Telescope* model, int* status)
{
    if (*status || !h || !model) return;

    /* Check the model is not empty. */
    if (oskar_telescope_num_stations(model) == 0)
    {
        oskar_log_error(h->log, "Telescope model is empty.");
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    /* Remove any existing telescope model, and copy the new one. */
    oskar_telescope_free(h->tel, status);
    h->tel = oskar_telescope_create_copy(model, OSKAR_CPU, status);

    /* Analyse the telescope model. */
    oskar_telescope_analyse(h->tel, status);
    oskar_telescope_log_summary(h->tel, h->log, status);
}

void oskar_interferometer_set_output_vis_file(oskar_Interferometer* h,
        const char* filename)
{
    if (!filename) return;
    const int len = (int) strlen(filename);
    free(h->vis_name);
    h->vis_name = 0;
    if (len == 0) return;
    h->vis_name = (char*) calloc(1 + len, 1);
    strcpy(h->vis_name, filename);
}

void oskar_interferometer_set_output_measurement_set(oskar_Interferometer* h,
        const char* filename)
{
    if (!filename) return;
    const int len = (int) strlen(filename);
    free(h->ms_name);
    h->ms_name = 0;
    if (len == 0) return;
    h->ms_name = (char*) calloc(6 + len, 1);
    if ((len >= 3) && (
            !strcmp(&(filename[len-3]), ".MS") ||
            !strcmp(&(filename[len-3]), ".ms") ))
        strcpy(h->ms_name, filename);
    else
        sprintf(h->ms_name, "%s.MS", filename);
}

void oskar_interferometer_set_source_flux_range(oskar_Interferometer* h,
        double min_jy, double max_jy)
{
    h->source_min_jy = min_jy;
    h->source_max_jy = max_jy;
}

void oskar_interferometer_set_zero_failed_gaussians(oskar_Interferometer* h,
        int value)
{
    h->zero_failed_gaussians = value;
}

const oskar_VisHeader* oskar_interferometer_vis_header(oskar_Interferometer* h)
{
    return h->header;
}

#ifdef __cplusplus
}
#endif
