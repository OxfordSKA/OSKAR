/*
 * Copyright (c) 2011, The University of Oxford
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

#include "apps/lib/oskar_Settings.h"
#include "apps/lib/oskar_set_up_sky.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_set_up_visibilities.h"
#include "apps/lib/oskar_write_ms.h"
#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_interferometer.h"
#include "interferometry/oskar_SimTime.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_split.h"
#include "utility/oskar_exit.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_add.h"

#include <QtCore/QByteArray>
#include <QtCore/QTime>

#include <cstdio>
#include <cstdlib>
#include <cmath>

using std::min;

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    // Parse command line.
    if (argc != 2)
    {
        fprintf(stderr, "Usage: $ oskar_sim [settings file]\n");
        return EXIT_FAILURE;
    }

    // Load the settings file.
    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    const oskar_SimTime* times = settings.obs().sim_time();

    // Find out how many GPUs we have.
    int device_count = 0;
    int num_devices = settings.num_cuda_devices();
    error = (int)cudaGetDeviceCount(&device_count);
    if (device_count < num_devices)
    {
        fprintf(stderr, "ERROR: Found %i CUDA device(s), but %i specified.\n",
                device_count, num_devices);
        return EXIT_FAILURE;
    }

    // Construct sky and telescope.
    oskar_SkyModel* sky_cpu = oskar_set_up_sky(settings);
    oskar_TelescopeModel* telescope_cpu = oskar_set_up_telescope(settings);
    if (sky_cpu == NULL || telescope_cpu == NULL)
    {
        fprintf(stderr, "ERROR: Could not set up sky or telescope data.\n");
        return EXIT_FAILURE;
    }

    // Split the sky model into chunks.
    oskar_SkyModel* sky_chunk_cpu = NULL;
    int num_sky_chunks = 0;
    int max_sources_per_chunk = min(settings.max_sources_per_chunk(),
            (int)ceil((double)sky_cpu->num_sources / (double)num_devices));
    error = oskar_sky_model_split(&sky_chunk_cpu, &num_sky_chunks,
            max_sources_per_chunk, sky_cpu);
    if (error) oskar_exit(error);
    printf("** Splitting sky model of %i sources into %i chunks "
            "(max %i per chunk).\n", sky_cpu->num_sources, num_sky_chunks,
            max_sources_per_chunk);

    // Create the global visibility structure on the CPU.
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Visibilities* vis_global = oskar_set_up_visibilities(settings,
            telescope_cpu, complex_matrix);

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU).
    oskar_Mem* vis_acc  = (oskar_Mem*)malloc(num_devices * sizeof(oskar_Mem));
    oskar_Mem* vis_temp = (oskar_Mem*)malloc(num_devices * sizeof(oskar_Mem));
    int time_baseline = telescope_cpu->num_baselines() * times->num_vis_dumps;
    for (int i = 0; i < num_devices; ++i)
    {
        error = oskar_mem_init(&vis_acc[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true);
        if (error) oskar_exit(error);
        error = oskar_mem_init(&vis_temp[i], complex_matrix, OSKAR_LOCATION_CPU,
                time_baseline, true);
        if (error) oskar_exit(error);
        cudaSetDevice(settings.cuda_device_ids()[i]);
        cudaDeviceSynchronize();
    }

    // Set the number of host threads to use (one per GPU).
    omp_set_num_threads(num_devices);

    // Run the simulation.
    printf("\n== Starting simulation...\n");
    QTime timer;
    timer.start();
    int num_channels = settings.obs().num_channels();
    for (int c = 0; c < num_channels; ++c)
    {
        printf("\n<< Channel (%i / %i).\n", c + 1, num_channels);
        double freq = settings.obs().frequency(c);

        // Use OpenMP dynamic scheduling for loop over chunks.
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < num_sky_chunks; ++i)
        {
            // Get thread ID for this chunk.
            int thread_id = omp_get_thread_num();

            // Get device ID and device properties for this chunk.
            int device_id = settings.cuda_device_ids()[thread_id];
            cudaDeviceProp device_prop;
            cudaGetDeviceProperties(&device_prop, device_id);

            // Set the device to use for the chunk.
            error = cudaSetDevice(device_id);
            if (error) oskar_exit(error);
            printf("\n*** Sky chunk (%i / %i : %i sources), device[%i] (%s).\n",
                    i + 1, num_sky_chunks, sky_chunk_cpu[i].num_sources,
                    device_id, device_prop.name);

            // Run simulation for this chunk.
            error = oskar_interferometer(&(vis_temp[thread_id]),
                    &(sky_chunk_cpu[i]), telescope_cpu, times, freq);
            if (error) oskar_exit(error);
            error = oskar_mem_add(&(vis_acc[thread_id]),
                    &(vis_acc[thread_id]), &(vis_temp[thread_id]));
            if (error) oskar_exit(error);
        }
        #pragma omp barrier

        oskar_Mem vis_amp;
        vis_global->get_channel_amps(&vis_amp, c);

        // Accumulate into global vis structure.
        for (int i = 0; i < num_devices; ++i)
        {
            error = oskar_mem_add(&vis_amp, &vis_amp, &vis_acc[i]);
            if (error) oskar_exit(error);
        }
    }

    // Add visibility noise.
    if (settings.sky().noise_model().toUpper() == "VLA_MEMO_146")
    {
        printf("== Adding Gaussian visibility noise.\n");
        error = vis_global->evaluate_sky_noise_stddev(telescope_cpu,
                settings.sky().noise_spectral_index());
        if (error) oskar_exit(error);
        error = vis_global->add_sky_noise(vis_global->sky_noise_stddev,
            settings.sky().noise_seed());
        if (error) oskar_exit(error);
    }

    printf("\n== Simulation completed in %f seconds.\n", timer.elapsed() / 1e3);

    // Compute baseline u,v,w coordinates for simulation.
    error = oskar_evaluate_baseline_uvw(vis_global, telescope_cpu, times);
    if (error) oskar_exit(error);

    // Write global visibilities to disk.
    if (!settings.obs().oskar_vis_filename().isEmpty())
    {
        QByteArray outname = settings.obs().oskar_vis_filename().toAscii();
        printf("\n--> Writing visibility file: '%s'\n", outname.constData());
        error = vis_global->write(outname);
        if (error) oskar_exit(error);
    }

#ifndef OSKAR_NO_MS
    // Write Measurement Set.
    if (!settings.obs().ms_filename().isEmpty())
    {
        QByteArray outname = settings.obs().ms_filename().toAscii();
        printf("--> Writing Measurement Set: '%s'\n", outname.constData());
        error = oskar_write_ms(outname, vis_global, telescope_cpu, true);
        if (error) oskar_exit(error);
    }
#endif

    // Delete data structures.
    delete vis_global;
    delete sky_cpu;
    delete telescope_cpu;
    for (int i = 0; i < num_devices; ++i)
    {
        error = oskar_mem_free(&vis_acc[i]);  if (error) oskar_exit(error);
        error = oskar_mem_free(&vis_temp[i]); if (error) oskar_exit(error);
        cudaSetDevice(settings.cuda_device_ids()[i]);
        cudaDeviceReset();
    }
    free(vis_acc);
    free(vis_temp);
    free(sky_chunk_cpu);

    return EXIT_SUCCESS;
}
