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
#include "apps/lib/oskar_Settings.h"
#include "apps/lib/oskar_settings_free.h"
#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_sim_beam_pattern.h"
#include "interferometry/oskar_SettingsTime.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "math/oskar_linspace.h"
#include "math/oskar_meshgrid.h"
#include "math/oskar_sph_from_lm.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_mjd_to_gast_fast.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "station/oskar_evaluate_station_beam.h"
#include "utility/oskar_Mem.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <QtCore/QByteArray>
#include <QtCore/QTime>

int oskar_sim_beam_pattern(const char* settings_file)
{
    int err;

    // Load the settings file.
    oskar_Settings settings;
    err = oskar_settings_load(&settings, settings_file);
    if (err) return err;
    const oskar_SettingsTime* times = &settings.obs.time;

    // Get the sky model and telescope model and copy both to GPU (slow step).
    oskar_TelescopeModel* tel_cpu, *tel_gpu;
    tel_cpu = oskar_set_up_telescope(&settings);
    tel_gpu = new oskar_TelescopeModel(tel_cpu, OSKAR_LOCATION_GPU);

    // Get the type.
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Get the image settings.
    int image_size = (int)settings.image.size;
    int num_channels = settings.obs.num_channels;
    int num_pixels = image_size * image_size;
    double fov_deg = settings.image.fov_deg;
    double lm_max = sin(fov_deg * M_PI / 180.0);
    double ra0 = settings.obs.ra0_rad;
    double dec0 = settings.obs.dec0_rad;

    // Generate l,m grid and equatorial coordinates for beam pattern pixels.
    oskar_Mem lm(type, OSKAR_LOCATION_CPU, image_size);
    oskar_Mem l_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem m_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem RA_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem Dec_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    if (type == OSKAR_SINGLE)
    {
        oskar_linspace_f(lm, -lm_max, lm_max, image_size);
        oskar_meshgrid_f(l_cpu, m_cpu, lm, image_size, lm, image_size);
        oskar_sph_from_lm_f(num_pixels, ra0, dec0, l_cpu, m_cpu, RA_cpu, Dec_cpu);
    }
    else if (type == OSKAR_DOUBLE)
    {
        oskar_linspace_d(lm, -lm_max, lm_max, image_size);
        oskar_meshgrid_d(l_cpu, m_cpu, lm, image_size, lm, image_size);
        oskar_sph_from_lm_d(num_pixels, ra0, dec0, l_cpu, m_cpu, RA_cpu, Dec_cpu);
    }

    // Get time data.
    int num_vis_dumps        = times->num_vis_dumps;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;

    // Allocate memory big enough for data hyper-cube.
    int num_elements = num_pixels * num_vis_dumps * num_channels;
    oskar_Mem data(type, OSKAR_LOCATION_CPU);

    // Open the data file.
    if (!settings.image.filename)
    {
        fprintf(stderr, "ERROR: No image file specified.\n");
        return EXIT_FAILURE;
    }
    FILE* file = fopen(settings.image.filename, "w");
    if (file == NULL) return OSKAR_ERR_FILE_IO;

    // Loop over channels.
    QTime timer;
    timer.start();
    for (int c = 0; c < num_channels; ++c)
    {
        // Initialise the random number generator.
        oskar_Device_curand_state curand_state(tel_cpu->max_station_size);
        int seed = 0; // TODO get this from the settings file....
        curand_state.init(seed);

        // Get the channel frequency.
        printf("\n--> Simulating channel (%d / %d).\n", c + 1, num_channels);
        double frequency = settings.obs.start_frequency_hz +
                c * settings.obs.frequency_inc_hz;

        // Copy RA and Dec to GPU and allocate arrays for pixel direction cosines.
        oskar_Mem RA(&RA_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem Dec(&Dec_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem l(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem m(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem n(type, OSKAR_LOCATION_GPU, num_pixels);

        // Allocate weights work array and memory for the beam pattern.
        oskar_Mem weights(type | OSKAR_COMPLEX, OSKAR_LOCATION_GPU);
        oskar_Mem beam_pattern(type | OSKAR_COMPLEX, OSKAR_LOCATION_GPU, num_pixels);

        // Copy the telescope model and scale coordinates to wavenumbers.
        oskar_TelescopeModel telescope(tel_gpu, OSKAR_LOCATION_GPU);
        err = telescope.multiply_by_wavenumber(frequency);
        if (err) return err;

        // Get pointer to the station.
        // FIXME Currently station 0: Determine this from the settings file?
        oskar_StationModel* station = &(telescope.station[0]);

        // Start simulation.
        for (int j = 0; j < num_vis_dumps; ++j)
        {
            // Start time for the visibility dump, in MJD(UTC).
            printf("--> Generating beam for snapshot (%i / %i).\n", j+1, num_vis_dumps);
            double t_dump = obs_start_mjd_utc + j * dt_dump;
            double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

            // Evaluate horizontal l,m,n for beam phase centre.
            double beam_l, beam_m, beam_n;
            err = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                    &beam_n, station, gast);
            if (err) return err;

            // Evaluate horizontal l,m,n coordinates.
            err = oskar_evaluate_source_horizontal_lmn(&l, &m, &n, &RA, &Dec,
                    station, gast);
            if (err) return err;

            // Evaluate the station beam.
            err = oskar_evaluate_station_beam(&beam_pattern, station, beam_l,
                    beam_m, &l, &m, &n, &weights, &curand_state);
            if (err) return err;

            // Copy beam pattern back to CPU.
            oskar_Mem beam_pattern_cpu(&beam_pattern, OSKAR_LOCATION_CPU);

            // Save beam to file for plotting.
            if (type == OSKAR_SINGLE)
            {
                for (int i = 0; i < num_pixels; ++i)
                {
                    fprintf(file, "%10.3f,%10.3f,%10.3f,%10.3f\n",
                            ((float*)l_cpu)[i],
                            ((float*)m_cpu)[i],
                            ((float2*)beam_pattern_cpu)[i].x,
                            ((float2*)beam_pattern_cpu)[i].y);
                }
            }
            else if (type == OSKAR_DOUBLE)
            {
                for (int i = 0; i < num_pixels; ++i)
                {
                    fprintf(file, "%10.3f,%10.3f,%10.3f,%10.3f\n",
                            ((double*)l_cpu)[i],
                            ((double*)m_cpu)[i],
                            ((double2*)beam_pattern_cpu)[i].x,
                            ((double2*)beam_pattern_cpu)[i].y);
                }
            }

            /*------------------------------------------------
            data = dlmread('temp_test_beam_pattern.txt');
            imagesc(log10(reshape(data(:,3), 401, 401).^2));
            --------------------------------------------------*/
        }
    }
    printf("=== Simulation completed in %f sec.\n", timer.elapsed() / 1.0e3);
    fclose(file);

    // Delete data structures.
    delete tel_gpu;
    delete tel_cpu;
    cudaDeviceReset();

    oskar_settings_free(&settings);
    return OSKAR_SUCCESS;
}
