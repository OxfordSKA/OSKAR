/*
 * Copyright (c) 2016-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_BEAM_PATTERN_ACCESSORS_H_
#define OSKAR_BEAM_PATTERN_ACCESSORS_H_

/**
 * @file oskar_beam_pattern_accessors.h
 */

#include <oskar_global.h>
#include <log/oskar_log.h>
#include <telescope/oskar_telescope.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
oskar_Log* oskar_beam_pattern_log(oskar_BeamPattern* h);

OSKAR_EXPORT
int oskar_beam_pattern_num_gpus(const oskar_BeamPattern* h);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_fits(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_phase_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_real_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_imag_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_text(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_auto_power_hdf5(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_average_time_and_channel(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_average_single_axis(oskar_BeamPattern* h,
        char option);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_amp_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_amp_text(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_amp_hdf5(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_phase_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_phase_text(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_phase_hdf5(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_real_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_imag_fits(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_raw_text(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_cross_power_raw_hdf5(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_coordinate_frame(oskar_BeamPattern* h, char option);

OSKAR_EXPORT
void oskar_beam_pattern_set_coordinate_type(oskar_BeamPattern* h, char option);

OSKAR_EXPORT
void oskar_beam_pattern_set_gpus(oskar_BeamPattern* h, int num_gpus,
        const int* cuda_device_ids, int* status);

OSKAR_EXPORT
void oskar_beam_pattern_set_image_size(oskar_BeamPattern* h,
        int width, int height);

OSKAR_EXPORT
void oskar_beam_pattern_set_image_cellsize(oskar_BeamPattern* h,
        double cellsize_arcsec);

OSKAR_EXPORT
void oskar_beam_pattern_set_image_fov(oskar_BeamPattern* h,
        double width_deg, double height_deg);

OSKAR_EXPORT
void oskar_beam_pattern_set_max_chunk_size(oskar_BeamPattern* h, int value);

OSKAR_EXPORT
void oskar_beam_pattern_set_num_devices(oskar_BeamPattern* h, int value);

OSKAR_EXPORT
void oskar_beam_pattern_set_observation_frequency(oskar_BeamPattern* h,
        double start_hz, double inc_hz, int num_channels);

OSKAR_EXPORT
void oskar_beam_pattern_set_observation_time(oskar_BeamPattern* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps);

OSKAR_EXPORT
void oskar_beam_pattern_set_root_path(oskar_BeamPattern* h, const char* path);

OSKAR_EXPORT
void oskar_beam_pattern_set_separate_time_and_channel(oskar_BeamPattern* h,
        int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_sky_model_file(oskar_BeamPattern* h,
        const char* path);

OSKAR_EXPORT
void oskar_beam_pattern_set_station_ids(oskar_BeamPattern* h,
        int num_stations, const int* ids);

OSKAR_EXPORT
void oskar_beam_pattern_set_test_source_stokes_i(oskar_BeamPattern* h,
        int enabled);

OSKAR_EXPORT
void oskar_beam_pattern_set_test_source_stokes_custom(oskar_BeamPattern* h,
        int enabled, double i, double q, double u, double v, int* status);

OSKAR_EXPORT
void oskar_beam_pattern_set_telescope_model(oskar_BeamPattern* h,
        const oskar_Telescope* model, int* status);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_amp_fits(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_amp_text(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_amp_hdf5(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_phase_fits(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_phase_text(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_phase_hdf5(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_raw_text(oskar_BeamPattern* h, int flag);

OSKAR_EXPORT
void oskar_beam_pattern_set_voltage_raw_hdf5(oskar_BeamPattern* h, int flag);


#ifdef __cplusplus
}
#endif

#endif /* include guard */
