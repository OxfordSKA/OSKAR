/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#ifndef OSKAR_INTERFEROMETER_H_
#define OSKAR_INTERFEROMETER_H_

/**
 * @file oskar_interferometer.h
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

struct oskar_Interferometer;
#ifndef OSKAR_INTERFEROMETER_TYPEDEF_
#define OSKAR_INTERFEROMETER_TYPEDEF_
typedef struct oskar_Interferometer oskar_Interferometer;
#endif

OSKAR_EXPORT
void oskar_interferometer_check_init(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
int oskar_interferometer_coords_only(const oskar_Interferometer* h);

OSKAR_EXPORT
oskar_Interferometer* oskar_interferometer_create(int precision, int* status);

OSKAR_EXPORT
oskar_VisBlock* oskar_interferometer_finalise_block(oskar_Interferometer* h,
        int block_index, int* status);

OSKAR_EXPORT
void oskar_interferometer_finalise(oskar_Interferometer* h, int* status);

OSKAR_EXPORT
void oskar_interferometer_free(oskar_Interferometer* h, int* status);

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
void oskar_interferometer_run_block(oskar_Interferometer* h, int block_index,
        int gpu_id, int* status);

OSKAR_EXPORT
void oskar_interferometer_run(oskar_Interferometer* h, int* status);

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
void oskar_interferometer_set_log(oskar_Interferometer* h, oskar_Log* log);

OSKAR_EXPORT
void oskar_interferometer_set_max_sources_per_chunk(oskar_Interferometer* h,
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

OSKAR_EXPORT
void oskar_interferometer_write_block(oskar_Interferometer* h,
        const oskar_VisBlock* block, int block_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_INTERFEROMETER_H_ */
