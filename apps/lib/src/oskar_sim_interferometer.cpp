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
#include "apps/lib/oskar_write_ms.h"
#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_interferometer.h"
#include "interferometry/oskar_SettingsTime.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_SettingsSky.h"
#include "sky/oskar_sky_model_free.h"
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

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using std::vector;

extern "C"
int oskar_sim_interferometer(const char* settings_file)
{
    int error;

    // Load the settings file.
    oskar_Settings settings;
    error = oskar_settings_load(&settings, settings_file);
    if (error) return error;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    const oskar_SettingsTime* times = &settings.obs.time;

    // Check that a data file has been specified.
    if (!settings.obs.oskar_vis_filename &&
            !settings.obs.ms_filename)
    {
        fprintf(stderr, "ERROR: No visibility file specified.\n");
        return OSKAR_ERR_SETTINGS;
    }

    // Find out how many GPUs we have.
    int device_count = 0;
    int num_devices = settings.sim.num_cuda_devices;
    error = (int)cudaGetDeviceCount(&device_count);
    if (error) return error;
    if (device_count < num_devices) return OSKAR_ERR_CUDA_DEVICES;

    // Setup the telescope model.
    oskar_TelescopeModel telescope_cpu;
    error = oskar_set_up_telescope(&telescope_cpu, &settings);
    if (error) return OSKAR_ERR_SETUP_FAIL;

    // Setup the sky model array.
    oskar_SkyModel* sky_chunk_cpu = NULL;
    int num_sky_chunks = 0;
    error = oskar_set_up_sky(&num_sky_chunks, &sky_chunk_cpu, &settings);
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
    int time_baseline = telescope_cpu.num_baselines() * times->num_time_steps;
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
    printf("\n=== Starting simulation...\n");
    QTime timer;
    timer.start();
    int num_channels = settings.obs.num_channels;
    for (int c = 0; c < num_channels; ++c)
    {
        double freq = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;
        printf("\n<< Channel (%i / %i) [%.4f MHz].\n", c + 1, num_channels, freq/1e6);

        // Use OpenMP dynamic scheduling for loop over chunks.
#pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_sky_chunks; ++i)
        {
            if (error) continue;

            // Get thread ID for this chunk.
            int thread_id = omp_get_thread_num();

            // Get device ID and device properties for this chunk.
            int device_id = settings.sim.cuda_device_ids[thread_id];
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, device_id);

            // Set the device to use for the chunk.
            error = cudaSetDevice(device_id);
            if (error) continue;
            printf("\n*** Sky chunk (%i / %i : %i sources), device[%i] (%s).\n",
                    i + 1, num_sky_chunks, sky_chunk_cpu[i].num_sources,
                    device_id, device_prop.name);

            // Run simulation for this chunk.
            error = oskar_interferometer(&(vis_temp[thread_id]),
                    &(sky_chunk_cpu[i]), &telescope_cpu, times, freq);
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

    printf("\n=== Simulation completed in %.3f sec.\n", timer.elapsed() / 1e3);

    // Compute baseline u,v,w coordinates for simulation.
    error = oskar_evaluate_baseline_uvw(&vis_global, &telescope_cpu, times);
    if (error) return error;

    // Write global visibilities to disk.
    if (settings.obs.oskar_vis_filename)
    {
        printf("\n--> Writing visibility file: '%s'\n",
                settings.obs.oskar_vis_filename);
        error = vis_global.write(settings.obs.oskar_vis_filename);
        if (error) return error;
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    if (settings.obs.ms_filename)
    {
        printf("--> Writing Measurement Set: '%s'\n", settings.obs.ms_filename);
        error = oskar_write_ms(settings.obs.ms_filename, &vis_global,
                &telescope_cpu, true);
        if (error) return error;
    }
#endif

    // Make image(s) of the simulated visibilities if required.
    if (settings.obs.image_interferometer_output)
    {
        if (settings.image.oskar_image || settings.image.fits_image)
        {
            oskar_Image image;
            printf("\n=== Starting OSKAR imager...\n");
            error = oskar_make_image(&image, &vis_global, &settings.image);
            printf("=== Imaging complete.\n\n");
            if (error) return error;
            if (settings.image.oskar_image)
            {
                printf("--> Writing OSKAR image: '%s'\n",
                        settings.image.oskar_image);
                error = oskar_image_write(&image, settings.image.oskar_image, 0);
                if (error) return error;
            }
#ifndef OSKAR_NO_FITS
            if (settings.image.fits_image)
            {
                printf("--> Writing FITS image: '%s'\n",
                        settings.image.fits_image);
                oskar_fits_image_write(&image, settings.image.fits_image);
            }
#endif
        }
        else
        {
            fprintf(stderr, "= WARNING: No image output name specified "
                    "(skipping OSKAR imager)\n");
        }
    }

    // Reset all CUDA devices.
    for (int i = 0; i < num_devices; ++i)
    {
        cudaSetDevice(settings.sim.cuda_device_ids[i]);
        cudaDeviceReset();
    }

    // FIXME Free sky chunks. This needs fixing in order to avoid
    // potential memory leaks (free memory using a destructor instead).
    for (int i = 0; i < num_sky_chunks; ++i)
        oskar_sky_model_free(&sky_chunk_cpu[i]);
    free(sky_chunk_cpu);

    fprintf(stdout, "\n=== Run complete.\n");
    return OSKAR_SUCCESS;
}
