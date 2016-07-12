/*
 * Copyright (c) 2012-2016, The University of Oxford
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

#ifndef OSKAR_SIM_INTERFEROMETER_H_
#define OSKAR_SIM_INTERFEROMETER_H_

/**
 * @file oskar_sim_interferometer.h
 */

#include <oskar_global.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_telescope.h>
#include <oskar_vis_block.h>

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Simulator;
#ifndef OSKAR_SIMULATOR_TYPEDEF_
#define OSKAR_SIMULATOR_TYPEDEF_
typedef struct oskar_Simulator oskar_Simulator;
#endif

OSKAR_APPS_EXPORT
void oskar_simulator_check_init(oskar_Simulator* h, int* status);

OSKAR_APPS_EXPORT
oskar_Simulator* oskar_simulator_create(int precision, int* status);

OSKAR_APPS_EXPORT
oskar_VisBlock* oskar_simulator_finalise_block(oskar_Simulator* h,
        int block_index, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_finalise(oskar_Simulator* h, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_free(oskar_Simulator* h, int* status);

OSKAR_APPS_EXPORT
int oskar_simulator_num_devices(const oskar_Simulator* h);

OSKAR_APPS_EXPORT
int oskar_simulator_num_gpus(const oskar_Simulator* h);

OSKAR_APPS_EXPORT
int oskar_simulator_num_vis_blocks(const oskar_Simulator* h);

OSKAR_APPS_EXPORT
void oskar_simulator_reset_cache(oskar_Simulator* h, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_reset_work_unit_index(oskar_Simulator* h);

OSKAR_APPS_EXPORT
void oskar_simulator_run_block(oskar_Simulator* h, int block_index,
        int gpu_id, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_run(oskar_Simulator* h, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_coords_only(oskar_Simulator* h, int value,
        int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_correlation_type(oskar_Simulator* h,
        const char* type, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_force_polarised_ms(oskar_Simulator* h, int value);

OSKAR_APPS_EXPORT
void oskar_simulator_set_gpus(oskar_Simulator* h, int num_gpus,
        int* cuda_device_ids, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_horizon_clip(oskar_Simulator* h, int value);

OSKAR_APPS_EXPORT
void oskar_simulator_set_log(oskar_Simulator* h, oskar_Log* log);

OSKAR_APPS_EXPORT
void oskar_simulator_set_max_times_per_block(oskar_Simulator* h, int value);

OSKAR_APPS_EXPORT
void oskar_simulator_set_num_devices(oskar_Simulator* h, int value);

OSKAR_APPS_EXPORT
void oskar_simulator_set_observation_frequency(oskar_Simulator* h,
        double start_hz, double inc_hz, int num_channels);

OSKAR_APPS_EXPORT
void oskar_simulator_set_observation_time(oskar_Simulator* h,
        double time_start_mjd_utc, double inc_sec, int num_time_steps);

OSKAR_APPS_EXPORT
void oskar_simulator_set_output_measurement_set(oskar_Simulator* h,
        const char* filename);

OSKAR_APPS_EXPORT
void oskar_simulator_set_output_vis_file(oskar_Simulator* h,
        const char* filename);

OSKAR_APPS_EXPORT
void oskar_simulator_set_settings_path(oskar_Simulator* h,
        const char* filename);

OSKAR_APPS_EXPORT
void oskar_simulator_set_sky_model(oskar_Simulator* h, const oskar_Sky* sky,
        int max_sources_per_chunk, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_telescope_model(oskar_Simulator* h,
        const oskar_Telescope* model, int* status);

OSKAR_APPS_EXPORT
void oskar_simulator_set_source_flux_range(oskar_Simulator* h,
        double min_jy, double max_jy);

OSKAR_APPS_EXPORT
void oskar_simulator_set_zero_failed_gaussians(oskar_Simulator* h, int value);

OSKAR_APPS_EXPORT
void oskar_simulator_write_block(oskar_Simulator* h,
        const oskar_VisBlock* block, int block_index, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SIM_INTERFEROMETER_H_ */
