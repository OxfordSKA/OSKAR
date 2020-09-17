/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_INTERFEROMETER_ACCESSORS_H_
#define OSKAR_INTERFEROMETER_ACCESSORS_H_

/**
 * @file oskar_interferometer_accessors.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>
#include <sky/oskar_sky.h>
#include <telescope/oskar_telescope.h>
#include <vis/oskar_vis_block.h>
#include <vis/oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
int oskar_interferometer_coords_only(const oskar_Interferometer* h);

OSKAR_EXPORT
oskar_Log* oskar_interferometer_log(oskar_Interferometer* h);

OSKAR_EXPORT
int oskar_interferometer_num_devices(const oskar_Interferometer* h);

OSKAR_EXPORT
int oskar_interferometer_num_gpus(const oskar_Interferometer* h);

OSKAR_EXPORT
int oskar_interferometer_num_vis_blocks(const oskar_Interferometer* h);

OSKAR_EXPORT
void oskar_interferometer_reset_cache(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_reset_work_unit_index(oskar_Interferometer* h);

OSKAR_EXPORT
void oskar_interferometer_set_coords_only(oskar_Interferometer* h, int value,
        int* status);

OSKAR_EXPORT
void oskar_interferometer_set_correlation_type(oskar_Interferometer* h,
        const char* type, int* status);

OSKAR_EXPORT
void oskar_interferometer_set_force_polarised_ms(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
void oskar_interferometer_set_gpus(oskar_Interferometer* h, int num_gpus,
        const int* cuda_device_ids, int* status);

OSKAR_EXPORT
void oskar_interferometer_set_horizon_clip(oskar_Interferometer* h, int value);

OSKAR_EXPORT
void oskar_interferometer_set_ignore_w_components(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
void oskar_interferometer_set_max_sources_per_chunk(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
void oskar_interferometer_set_max_channels_per_block(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
void oskar_interferometer_set_max_times_per_block(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
void oskar_interferometer_set_num_devices(oskar_Interferometer* h, int value);

OSKAR_EXPORT
void oskar_interferometer_set_observation_frequency(oskar_Interferometer* h,
        double start_hz, double inc_hz, int num_channels);

OSKAR_EXPORT
void oskar_interferometer_set_observation_time(oskar_Interferometer* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps);

OSKAR_EXPORT
void oskar_interferometer_set_output_measurement_set(oskar_Interferometer* h,
        const char* filename);

OSKAR_EXPORT
void oskar_interferometer_set_output_vis_file(oskar_Interferometer* h,
        const char* filename);

OSKAR_EXPORT
void oskar_interferometer_set_settings_path(oskar_Interferometer* h,
        const char* filename);

OSKAR_EXPORT
void oskar_interferometer_set_sky_model(oskar_Interferometer* h,
        const oskar_Sky* sky, int* status);

OSKAR_EXPORT
void oskar_interferometer_set_telescope_model(oskar_Interferometer* h,
        const oskar_Telescope* model, int* status);

OSKAR_EXPORT
void oskar_interferometer_set_source_flux_range(oskar_Interferometer* h,
        double min_jy, double max_jy);

OSKAR_EXPORT
void oskar_interferometer_set_zero_failed_gaussians(oskar_Interferometer* h,
        int value);

OSKAR_EXPORT
const oskar_VisHeader* oskar_interferometer_vis_header(oskar_Interferometer* h);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
