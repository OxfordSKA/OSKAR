/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <cuda_runtime_api.h>
#include <omp.h>

#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_set_up_sky.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_set_up_visibilities.h"
#include "apps/lib/oskar_sim_interferometer.h"
#include "apps/lib/oskar_vis_write_ms.h"

#include <oskar_evaluate_uvw_baseline.h>
#include <oskar_interferometer.h>
#include <oskar_log.h>
#include <oskar_sky.h>
#include <oskar_settings_free.h>
#include <oskar_telescope.h>
#include <oskar_timers.h>
#include <oskar_vis.h>
#include <oskar_make_image.h>
#include <oskar_image_write.h>
#include <oskar_image_free.h>
#ifndef OSKAR_NO_FITS
#include <fits/oskar_fits_image_write.h>
#endif

#include <cstdlib>
#include <cmath>
#include <vector>

using std::vector;

static void make_image(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, oskar_Log* log, int* status);

static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log);

extern "C"
int oskar_sim_interferometer(const char* settings_file, oskar_Log* log)
{
    int error;
    const char* fname;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    error = oskar_settings_load(&settings, log, settings_file);
    if (error) return error;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_sky(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_interferometer(log, &settings);
    //oskar_log_settings_ionosphere(log, &settings);

    if (settings.interferometer.image_interferometer_output)
        oskar_log_settings_image(log, &settings);

    // Check that a data file has been specified.
    if ( !(settings.interferometer.oskar_vis_filename ||
            settings.interferometer.ms_filename ||
            (settings.interferometer.image_interferometer_output &&
                    (settings.image.oskar_image || settings.image.fits_image))))
    {
        oskar_log_error(log, "No output file specified.");
        return OSKAR_ERR_SETTINGS;
    }

    // Find out how many GPUs we have, initialise and create timers for each.
    int num_devices, device_count = 0;
    num_devices = settings.sim.num_cuda_devices;
    error = (int)cudaGetDeviceCount(&device_count);
    if (error) return error;
    if (device_count < num_devices) return OSKAR_ERR_CUDA_DEVICES;
    vector<oskar_Timers> timers(num_devices);
    for (int i = 0; i < num_devices; ++i)
    {
        error = (int)cudaSetDevice(settings.sim.cuda_device_ids[i]);
        if (error) return error;
        cudaDeviceSynchronize();
        oskar_timers_create(&timers[i], OSKAR_TIMER_CUDA);
    }

    // Set up the telescope model.
    oskar_Telescope* tel = oskar_set_up_telescope(log, &settings, &error);
    if (error) return error;

    // Set up the sky model array.
    int num_sky_chunks = 0;
    oskar_Sky** sky_chunks = oskar_set_up_sky(&num_sky_chunks, log,
            &settings, &error);
    if (error) return error;

    // Create the global visibility structure on the CPU.
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Vis* vis = oskar_set_up_visibilities(&settings, tel, complex_matrix,
            &error);

    // Must check for errors to ensure there are no null pointers.
    if (error) return error;

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU).
    vector<oskar_Mem> vis_acc(num_devices), vis_temp(num_devices);
    int time_baseline = oskar_telescope_num_baselines(tel) *
            settings.obs.num_time_steps;
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_mem_init(&vis_acc[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true, &error);
        oskar_mem_init(&vis_temp[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true, &error);
    }

    // Set the number of host threads to use (one per GPU).
    omp_set_num_threads(num_devices);

    // Run the simulation.
    cudaSetDevice(settings.sim.cuda_device_ids[0]);
    oskar_log_section(log, "Starting simulation...");
    oskar_timer_start(timers[0].tmr);
    for (int c = 0; c < settings.obs.num_channels; ++c)
    {
        double frequency;
        oskar_Mem vis_amp;

        frequency = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;

        oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, settings.obs.num_channels, frequency / 1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_sky_chunks; ++i)
        {
            if (error) continue;

            // Get thread ID for this chunk, and set device for this thread.
            int thread_id = omp_get_thread_num();
            error = cudaSetDevice(settings.sim.cuda_device_ids[thread_id]);

            // Run simulation for this chunk.
            oskar_interferometer(&(vis_temp[thread_id]), log,
                    &timers[thread_id], sky_chunks[i], tel, &settings,
                    frequency, i, num_sky_chunks, &error);

            oskar_timer_resume(timers[thread_id].tmr_init_copy);
            oskar_mem_add(&(vis_acc[thread_id]), &(vis_acc[thread_id]),
                    &(vis_temp[thread_id]), &error);
            oskar_timer_pause(timers[thread_id].tmr_init_copy);
        }
#pragma omp barrier

        // Accumulate each chunk into global vis structure for this channel.
        oskar_vis_get_channel_amps(&vis_amp, vis, c, &error);
        for (int i = 0; i < num_devices; ++i)
        {
            cudaSetDevice(settings.sim.cuda_device_ids[i]);
            oskar_timer_resume(timers[i].tmr_init_copy);
            oskar_mem_add(&vis_amp, &vis_amp, &vis_acc[i], &error);

            // Clear thread accumulation buffer.
            oskar_mem_clear_contents(&vis_acc[i], &error);
            oskar_timer_pause(timers[i].tmr_init_copy);
        }
    }

    // Add uncorrelated system noise to the visibilities.
    if (settings.interferometer.noise.enable)
    {
        int seed = settings.interferometer.noise.seed;
        oskar_vis_add_system_noise(vis, tel, seed, &error);
    }

    // Free unneeded memory.
    for (int i = 0; i < num_devices; ++i)
    {
        oskar_mem_free(&vis_acc[i], &error);
        oskar_mem_free(&vis_temp[i], &error);
    }
    for (int i = 0; i < num_sky_chunks; ++i)
    {
        oskar_sky_free(sky_chunks[i], &error);
    }
    free(sky_chunks);
    oskar_telescope_free(tel, &error);

    // Record times.
    record_timing(num_devices, settings.sim.cuda_device_ids, &timers[0], log);

    // Write visibilities to disk.
    fname = settings.interferometer.oskar_vis_filename;
    if (fname && !error)
    {
        oskar_log_message(log, 0, "Writing OSKAR visibility file: '%s'", fname);
        oskar_vis_write(vis, log, fname, &error);
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    fname = settings.interferometer.ms_filename;
    if (fname && !error)
    {
        oskar_log_message(log, 0, "Writing Measurement Set: '%s'", fname);
        oskar_vis_write_ms(vis, fname, true, &error);
    }
#endif

    // Make image(s) of the visibilities using first device, if required.
    if (settings.interferometer.image_interferometer_output)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[0]);
        make_image(vis, &settings.image, log, &error);
    }

    // Free visibility data.
    oskar_vis_free(vis, &error);

    // Reset all CUDA devices and destroy timers.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[i]);
        oskar_timers_free(&timers[i]);
        cudaDeviceReset();
    }

    if (!error)
        oskar_log_section(log, "Run complete.");
    return error;
}


static void make_image(const oskar_Vis* vis,
        const oskar_SettingsImage* settings, oskar_Log* log, int* status)
{
    oskar_Timer* tmr;
    oskar_Image image;
    const char* filename;

    if (*status) return;

    // Check filenames.
    if (!settings->oskar_image && !settings->fits_image)
    {
        oskar_log_warning(log, "No image output name specified "
                "(skipping OSKAR imager)");
        return;
    }

    // Make image(s).
    tmr = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_log_section(log, "Starting OSKAR imager...");
    oskar_timer_start(tmr);
    *status = oskar_make_image(&image, log, vis, settings);
    oskar_log_section(log, "Imaging completed in %.3f sec.",
            oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);

    // Write image file(s).
#ifndef OSKAR_NO_FITS
    filename = settings->fits_image;
    if (filename)
    {
        oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
        oskar_fits_image_write(&image, log, filename, status);
    }
#endif
    filename = settings->oskar_image;
    if (filename)
    {
        oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(&image, log, filename, 0, status);
    }
    oskar_image_free(&image, status);
}


static void record_timing(int num_devices, int* cuda_device_ids,
        oskar_Timers* timers, oskar_Log* log)
{
    double elapsed, t_init = 0.0, t_clip = 0.0, t_R = 0.0, t_E = 0.0, t_K = 0.0;
    double t_join = 0.0, t_correlate = 0.0;

    // Record time taken.
    cudaSetDevice(cuda_device_ids[0]);
    elapsed = oskar_timer_elapsed(timers[0].tmr);
    oskar_log_section(log, "Simulation completed in %.3f sec.", elapsed);

    // Record percentage times.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(cuda_device_ids[i]);
        t_init += oskar_timer_elapsed(timers[i].tmr_init_copy);
        t_clip += oskar_timer_elapsed(timers[i].tmr_clip);
        t_R += oskar_timer_elapsed(timers[i].tmr_R);
        t_E += oskar_timer_elapsed(timers[i].tmr_E);
        t_K += oskar_timer_elapsed(timers[i].tmr_K);
        t_join += oskar_timer_elapsed(timers[i].tmr_join);
        t_correlate += oskar_timer_elapsed(timers[i].tmr_correlate);
    }
    t_init *= (100.0 / (num_devices * elapsed));
    t_clip *= (100.0 / (num_devices * elapsed));
    t_R *= (100.0 / (num_devices * elapsed));
    t_E *= (100.0 / (num_devices * elapsed));
    t_K *= (100.0 / (num_devices * elapsed));
    t_join *= (100.0 / (num_devices * elapsed));
    t_correlate *= (100.0 / (num_devices * elapsed));
    oskar_log_message(log, -1, "%4.1f%% Chunk copy & initialise.", t_init);
    oskar_log_message(log, -1, "%4.1f%% Horizon clip.", t_clip);
    oskar_log_message(log, -1, "%4.1f%% Jones R.", t_R);
    oskar_log_message(log, -1, "%4.1f%% Jones E.", t_E);
    oskar_log_message(log, -1, "%4.1f%% Jones K.", t_K);
    oskar_log_message(log, -1, "%4.1f%% Jones join.", t_join);
    oskar_log_message(log, -1, "%4.1f%% Jones correlate.", t_correlate);
    oskar_log_message(log, -1, "");
}
