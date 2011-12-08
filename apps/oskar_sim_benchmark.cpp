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
#include "apps/lib/oskar_write_ms.h"
#include "interferometry/oskar_evaluate_baseline_uvw.h"
#include "interferometry/oskar_interferometer.h"
#include "interferometry/oskar_SimTime.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_Visibilities.h"
#include "station/oskar_station_model_resize.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_split.h"
#include "utility/oskar_exit.h"
#include "utility/oskar_Mem.h"
#include "math/oskar_round_robin.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_add.h"
#include "utility/oskar_mem_clear_contents.h"
#include "utility/oskar_vector_types.h"

#include <QtCore/QByteArray>
#include <QtCore/QTime>

#include <cstdio>
#include <cstdlib>
#include <cmath>

using std::min;

oskar_SkyModel* oskar_set_up_benchmark_sky(const oskar_Settings& settings);
oskar_TelescopeModel* oskar_set_up_benchmark_telescope(const oskar_Settings& settings);

int main(int argc, char** argv)
{
    int error = OSKAR_SUCCESS;

    // Parse command line.
    if (argc != 2)
    {
        fprintf(stderr, "ERROR: Missing command line arguments.\n");
        fprintf(stderr, "Usage:  $ oskar_sim_benchmark [settings file]\n");
        return EXIT_FAILURE;
    }

    // Load the settings file.
    oskar_Settings settings;
    if (!settings.load(QString(argv[1]))) return EXIT_FAILURE;
    settings.print();
    const oskar_SimTime* times = settings.obs().sim_time();

    // Construct sky and telescope structures on the CPU.
    // NOTE replace these to load real data...
    oskar_SkyModel* sky_cpu;
//    sky_cpu = oskar_set_up_benchmark_sky(settings);
    sky_cpu = oskar_set_up_sky(settings);
    if (sky_cpu == NULL) oskar_exit(OSKAR_ERR_UNKNOWN);
    oskar_TelescopeModel* telescope_cpu;
//    telescope_cpu = oskar_set_up_benchmark_telescope(settings);
    telescope_cpu = oskar_set_up_telescope(settings);
    if (telescope_cpu == NULL) oskar_exit(OSKAR_ERR_UNKNOWN);

    // Split the sky model into chunks.
    oskar_SkyModel* sky_chunk_cpu = NULL;
    int max_sources_per_gpu = settings.max_sources_per_gpu();
    int num_sky_chunks = 0;
    error = oskar_sky_model_split(&sky_chunk_cpu, &num_sky_chunks,
            max_sources_per_gpu, sky_cpu);
    if (error) oskar_exit(error);
    printf("* Input sky model of %i sources will be split into %i chunks.\n\n",
            sky_cpu->num_sources, num_sky_chunks);

    // Create the global visibility structure on the CPU
    int num_channels = settings.obs().num_channels();
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int complex_matrix = type | OSKAR_COMPLEX | OSKAR_MATRIX;
    oskar_Visibilities vis_global(complex_matrix, OSKAR_LOCATION_CPU,
            num_channels, times->num_vis_dumps, telescope_cpu->num_baselines());
    // Add visibility meta-data.
    vis_global.freq_start_hz      = settings.obs().start_frequency();
    vis_global.freq_inc_hz        = settings.obs().frequency_inc();
    vis_global.time_start_mjd_utc = times->obs_start_mjd_utc;
    vis_global.time_inc_seconds   = times->dt_dump_days * 86400.0;


    // Find out how many GPU's we have.
    int device_count = 0;
    error = (int)cudaGetDeviceCount(&device_count);
    printf("== Found %i CUDA devices!\n", device_count);
    if (device_count < (int)settings.num_devices())
    {
        fprintf(stderr, "ERROR: Only found %i devices, %i specified!\n",
                device_count, settings.num_devices());
        oskar_exit(OSKAR_ERR_UNKNOWN);
    }

    // Set the number of host threads to use.
    int num_omp_threads = min(device_count, (int)settings.max_host_threads());
    omp_set_num_threads(num_omp_threads);

    // Create temporary and accumulation buffers to hold visibility amplitudes
    // (one per thread/GPU).
    oskar_Mem* vis_acc = NULL;
    oskar_Mem* vis_temp = NULL;
    vis_acc  = (oskar_Mem*)malloc(settings.num_devices() * sizeof(oskar_Mem));
    vis_temp = (oskar_Mem*)malloc(settings.num_devices() * sizeof(oskar_Mem));
    int num_elements = telescope_cpu->num_baselines() * times->num_vis_dumps;
    printf("num_elements = %i\n", num_elements);
    for (int i = 0; i < (int)settings.num_devices(); ++i)
    {
        error = oskar_mem_init(&vis_acc[i], complex_matrix, OSKAR_LOCATION_CPU,
                telescope_cpu->num_baselines() * times->num_vis_dumps, OSKAR_TRUE);
        if (error) oskar_exit(error);
        error = oskar_mem_clear_contents(&vis_acc[i]);
        if (error) oskar_exit(error);

        error = oskar_mem_init(&vis_temp[i], complex_matrix, OSKAR_LOCATION_CPU,
                telescope_cpu->num_baselines() * times->num_vis_dumps, OSKAR_TRUE);
        if (error) oskar_exit(error);
        error = oskar_mem_clear_contents(&vis_temp[i]);
        if (error) oskar_exit(error);
    }

    // ################## SIMULATION ###########################################
    printf("\nStarting simulation ...\n");
    for (int c = 0; c < num_channels; ++c)
    {
        printf(" --> channel %i (of %i)\n", c + 1, num_channels);
        double freq = settings.obs().frequency(c);

        #pragma omp parallel shared(vis_acc, vis_temp)
        {
            int num_threads = omp_get_num_threads();
            int thread_id  = omp_get_thread_num();
            int num_chunks = 0, start_chunk = 0;
            oskar_round_robin(num_sky_chunks, num_threads, thread_id,
                    &num_chunks, &start_chunk);
            int device_id = settings.use_devices()[thread_id];

            printf("\t== omp thread id = %i (of %i) (using device id %i) to process sky "
                    "chunks %i to %i\n", thread_id, num_threads, device_id,
                    start_chunk, start_chunk + num_chunks -1);

            error = cudaSetDevice(device_id);
            if (error) oskar_exit(error);

            for (int i = start_chunk; i <  start_chunk + num_chunks; ++i)
            {
                printf("<<==== Sky chunk %i ====>>\n", i);
                 error = oskar_interferometer(&(vis_temp[thread_id]),
                        &(sky_chunk_cpu[i]), telescope_cpu, times, freq);
                if (error) oskar_exit(error);

                 printf("--> Accumulating visibilities\n");
                error = oskar_mem_add(&(vis_acc[thread_id]), &(vis_acc[thread_id]),
                        &(vis_temp[thread_id]));
                if (error) oskar_exit(error);
            }
        } // end of omp parallel region.

        oskar_Mem vis_amp;
        vis_global.get_channel_amps(&vis_amp, c);
        double4c* v = (double4c*)(vis_amp.data);

        // Accumulate into global vis structure.
        for (int t = 0; t < num_omp_threads; ++t)
        {
            error = oskar_mem_add(&vis_amp, &vis_amp, &vis_acc[t]);
            if (error) oskar_exit(error);
        }

    } // end loop over channels

    // Compute baseline u,v,w coordinates for simulation.
    error = oskar_evaluate_baseline_uvw(&vis_global, telescope_cpu, times);
    if (error) oskar_exit(error);



    // Write global visibilities to disk.
    if (!settings.obs().oskar_vis_filename().isEmpty())
    {
        QByteArray outname = settings.obs().oskar_vis_filename().toAscii();
        printf("--> Writing visibility file: '%s'\n", outname.constData());
        error = vis_global.write(outname);
        if (error) oskar_exit(error);
    }

    // Delete data structures.
    delete sky_cpu;
    delete telescope_cpu;
    for (int i = 0; i < (int)settings.num_devices(); ++i)
    {
        error = oskar_mem_free(&vis_acc[i]);
        if (error) oskar_exit(error);
        error = oskar_mem_free(&vis_temp[i]);
        if (error) oskar_exit(error);
    }
    free(vis_acc);
    free(vis_temp);
    free(sky_chunk_cpu);
    cudaDeviceReset();

    return EXIT_SUCCESS;
}


oskar_SkyModel* oskar_set_up_benchmark_sky(const oskar_Settings& settings)
{
    oskar_SkyModel* sky = NULL;

    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int num_sources = settings.benchmark().num_sources();

    // Declare a sky model structure.
    sky = new oskar_SkyModel(type, OSKAR_LOCATION_CPU, num_sources);

    return sky;
}


oskar_TelescopeModel* oskar_set_up_benchmark_telescope(const oskar_Settings& settings)
{
    oskar_TelescopeModel *telescope = NULL;

    int error = OSKAR_SUCCESS;
    int type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int num_stations = settings.benchmark().num_stations();
    int num_antennas = settings.benchmark().num_antennas();

    // Declare a telescope structure.
    telescope = new oskar_TelescopeModel(type, OSKAR_LOCATION_CPU, num_stations);

    for (int i = 0; i < num_stations; ++i)
    {
        error = oskar_station_model_resize(&telescope->station[i], num_antennas);
        if (error) oskar_exit(error);
    }

    telescope->identical_stations = OSKAR_FALSE;
    telescope->coord_units = OSKAR_METRES;
    telescope->use_common_sky = OSKAR_TRUE;
    telescope->ra0_rad = 0.0;
    telescope->dec0_rad = 0.0;
    telescope->wavelength_metres = 0.0;
    telescope->bandwidth_hz = 0.0;

    return telescope;
}

