/*
 * Copyright (c) 2012, The University of Oxford
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
#include "apps/lib/oskar_visibilities_write_ms.h"
#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_interferometer.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_write.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_SettingsSky.h"
#include "sky/oskar_sky_model_free.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_settings.h"
#include "utility/oskar_log_warning.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_add.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_settings_free.h"
#include "imaging/oskar_make_image.h"
#include "imaging/oskar_image_write.h"
#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_image_write.h"
#endif

#include <QtCore/QTime>

#include <cstdlib>
#include <cmath>
#include <vector>

using std::vector;

extern "C"
int oskar_sim_interferometer(const char* settings_file, oskar_Log* log)
{
    int error;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    error = oskar_settings_load(&settings, log, settings_file);
    if (error) return error;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Log the relevant settings.
    log->keep_file = settings.sim.keep_log_file;
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_sky(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_interferometer(log, &settings);
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

    // Find out how many GPUs we have.
    int device_count = 0;
    int num_devices = settings.sim.num_cuda_devices;
    error = (int)cudaGetDeviceCount(&device_count);
    if (error) return error;
    if (device_count < num_devices) return OSKAR_ERR_CUDA_DEVICES;

    // Set up the telescope model.
    oskar_TelescopeModel telescope_cpu;
    error = oskar_set_up_telescope(&telescope_cpu, log, &settings);
    if (error) return OSKAR_ERR_SETUP_FAIL;

    // Set up the sky model array.
    oskar_SkyModel* sky_chunk_cpu = NULL;
    int num_sky_chunks = 0;
    error = oskar_set_up_sky(&num_sky_chunks, &sky_chunk_cpu, log, &settings);
    if (error) return error;

    // Create the global visibility structure on the CPU.
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Visibilities vis_global;
    error = oskar_set_up_visibilities(&vis_global, &settings, &telescope_cpu,
            complex_matrix);
    if (error) return error;

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU).
    // These are held in standard vectors so that the memory will be released
    // automatically if the function returns early, or is terminated.
    vector<oskar_Mem> vis_acc(num_devices), vis_temp(num_devices);
    int time_baseline = telescope_cpu.num_baselines() * settings.obs.num_time_steps;
    for (int i = 0; i < num_devices; ++i)
    {
        error = oskar_mem_init(&vis_acc[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true);
        if (error) return error;
        error = oskar_mem_init(&vis_temp[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true);
        if (error) return error;
        error = cudaSetDevice(settings.sim.cuda_device_ids[i]);
        if (error) return error;
        cudaDeviceSynchronize();
    }

    // Set the number of host threads to use (one per GPU).
    omp_set_num_threads(num_devices);

    // Run the simulation.
    oskar_log_section(log, "Starting simulation...");
    QTime timer;
    timer.start();
    int num_channels = settings.obs.num_channels;
    for (int c = 0; c < num_channels; ++c)
    {
        double frequency = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;
        oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                c + 1, num_channels, frequency / 1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_sky_chunks; ++i)
        {
            if (error) continue;

            // Get thread ID for this chunk.
            int thread_id = omp_get_thread_num();

            // Get device ID and device properties for this chunk.
            int device_id = settings.sim.cuda_device_ids[thread_id];

            // Set the device to use for the chunk.
            error = cudaSetDevice(device_id);
            if (error) continue;

            // Run simulation for this chunk.
            error = oskar_interferometer(&(vis_temp[thread_id]), log,
                    &(sky_chunk_cpu[i]), &telescope_cpu, &settings, frequency,
                    i, num_sky_chunks);
            if (error) continue;

            error = oskar_mem_add(&(vis_acc[thread_id]),
                    &(vis_acc[thread_id]), &(vis_temp[thread_id]));
            if (error) continue;
        }
#pragma omp barrier
        if (error) return error;

        oskar_Mem vis_amp;
        error = vis_global.get_channel_amps(&vis_amp, c);
        if (error) return error;

        // Accumulate into global vis structure.
        for (int i = 0; i < num_devices; ++i)
        {
            error = oskar_mem_add(&vis_amp, &vis_amp, &vis_acc[i]);
            if (error) return error;

            // Clear thread accumulation buffer
            vis_acc[i].clear_contents();
        }
    }
    oskar_log_section(log, "Simulation completed in %.3f sec.",
            timer.elapsed() / 1e3);

    // Compute baseline u,v,w coordinates for simulation.
    error = oskar_evaluate_baseline_uvw(&vis_global, &telescope_cpu, &settings.obs);
    if (error) return error;

    // Write global visibilities to disk.
    if (settings.interferometer.oskar_vis_filename)
    {
        error = oskar_visibilities_write(&vis_global, log,
                settings.interferometer.oskar_vis_filename);
        if (error) return error;
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    if (settings.interferometer.ms_filename)
    {
        error = oskar_visibilities_write_ms(&vis_global, log,
                settings.interferometer.ms_filename, true);
        if (error) return error;
    }
#endif

    // Make image(s) of the simulated visibilities if required.
    if (settings.interferometer.image_interferometer_output)
    {
        if (settings.image.oskar_image || settings.image.fits_image)
        {
            oskar_Image image;
            oskar_log_section(log, "Starting OSKAR imager...");
            error = oskar_make_image(&image, log, &vis_global, &settings.image);
            oskar_log_section(log, "Imaging complete.");
            if (error) return error;
            if (settings.image.oskar_image)
            {
                error = oskar_image_write(&image, log,
                        settings.image.oskar_image, 0);
                if (error) return error;
            }
#ifndef OSKAR_NO_FITS
            if (settings.image.fits_image)
            {
                error = oskar_fits_image_write(&image, log,
                        settings.image.fits_image);
                if (error) return error;
            }
#endif
        }
        else
        {
            oskar_log_warning(log, "No image output name specified "
                    "(skipping OSKAR imager)");
        }
    }

    // Reset all CUDA devices.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[i]);
        cudaDeviceReset();
    }

    // FIXME Free sky chunks. This needs fixing in order to avoid potential
    // memory leaks in case of errors (free memory using a destructor instead).
    for (int i = 0; i < num_sky_chunks; ++i)
    {
        oskar_sky_model_free(&sky_chunk_cpu[i]);
    }
    free(sky_chunk_cpu);

    oskar_log_section(log, "Run complete.");
    return OSKAR_SUCCESS;
}
