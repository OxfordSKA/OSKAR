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
#include "fits/oskar_fits_image_write.h"
#include "interferometry/oskar_SettingsTime.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "imaging/oskar_evaluate_image_lm_grid.h"
#include "imaging/oskar_Image.h"
#include "imaging/oskar_image_resize.h"
#include "imaging/oskar_image_write.h"
#include "math/oskar_sph_from_lm.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_mjd_to_gast_fast.h"
#include "station/oskar_evaluate_beam_horizontal_lmn.h"
#include "station/oskar_evaluate_source_horizontal_lmn.h"
#include "station/oskar_evaluate_station_beam.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_copy.h"

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
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Check that a data file has been specified.
    if (!settings.beam_pattern.filename && !settings.beam_pattern.fits_image)
    {
        fprintf(stderr, "ERROR: No image file specified.\n");
        return OSKAR_ERR_SETTINGS;
    }

    // Get the telescope model.
    oskar_TelescopeModel* tel_cpu = oskar_set_up_telescope(&settings);

    // Get the beam pattern settings.
    int station_id = settings.beam_pattern.station_id;
    int image_size = settings.beam_pattern.size;
    int num_channels = settings.obs.num_channels;
    int num_pixels = image_size * image_size;
    double fov = settings.beam_pattern.fov_deg * M_PI / 180;
    double ra0 = settings.obs.ra0_rad;
    double dec0 = settings.obs.dec0_rad;

    // Check station ID is within range.
    if (station_id < 0 || station_id >= tel_cpu->num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;

    // Get time data.
    int num_times            = times->num_vis_dumps;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;

    // Declare image hyper-cube.
    oskar_Image data(type, OSKAR_LOCATION_CPU);
    err = oskar_image_resize(&data, image_size, image_size, 1,
            num_times, num_channels);
    if (err) return err;

    // Set image meta-data.
    data.centre_ra_deg      = settings.obs.ra0_rad * 180.0 / M_PI;
    data.centre_dec_deg     = settings.obs.dec0_rad * 180.0 / M_PI;
    data.fov_ra_deg         = settings.beam_pattern.fov_deg;
    data.fov_dec_deg        = settings.beam_pattern.fov_deg;
    data.freq_start_hz      = settings.obs.start_frequency_hz;
    data.freq_inc_hz        = settings.obs.frequency_inc_hz;
    data.time_inc_sec       = settings.obs.time.dt_dump_days * 86400.0;
    data.time_start_mjd_utc = settings.obs.time.obs_start_mjd_utc;

    // Temporary CPU memory.
    oskar_Mem beam_cpu(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU, num_pixels);

    // Generate l,m grid and equatorial coordinates for beam pattern pixels.
    oskar_Mem grid_l(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem grid_m(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem RA_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem Dec_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    if (type == OSKAR_SINGLE)
    {
        oskar_evaluate_image_lm_grid_f(image_size, image_size, fov, fov,
                grid_l, grid_m);
        oskar_sph_from_lm_f(num_pixels, ra0, dec0,
                grid_l, grid_m, RA_cpu, Dec_cpu);
    }
    else if (type == OSKAR_DOUBLE)
    {
        oskar_evaluate_image_lm_grid_d(image_size, image_size, fov, fov,
                grid_l, grid_m);
        oskar_sph_from_lm_d(num_pixels, ra0, dec0,
                grid_l, grid_m, RA_cpu, Dec_cpu);
    }

    // All GPU memory used within these braces.
    {
        // Copy telescope model to GPU.
        oskar_TelescopeModel tel_gpu(tel_cpu, OSKAR_LOCATION_GPU);

        // Copy RA and Dec to GPU and allocate arrays for direction cosines.
        oskar_Mem RA(&RA_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem Dec(&Dec_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem l(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem m(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem n(type, OSKAR_LOCATION_GPU, num_pixels);

        // Allocate weights work array and GPU memory for a beam pattern.
        oskar_Mem weights(type | OSKAR_COMPLEX, OSKAR_LOCATION_GPU);
        oskar_Mem beam_pattern(type | OSKAR_COMPLEX, OSKAR_LOCATION_GPU,
                num_pixels);

        // Loop over channels.
        QTime timer;
        timer.start();
        for (int c = 0; c < num_channels; ++c)
        {
            // Initialise the random number generator.
            oskar_Device_curand_state curand_state(tel_gpu.max_station_size);
            int seed = 0; // TODO get this from the settings file....
            curand_state.init(seed);

            // Get the channel frequency.
            printf("\n--> Simulating channel (%d / %d).\n", c+1, num_channels);
            double frequency = settings.obs.start_frequency_hz +
                    c * settings.obs.frequency_inc_hz;

            // Copy the telescope model and scale coordinates to radians.
            oskar_TelescopeModel telescope(&tel_gpu, OSKAR_LOCATION_GPU);
            err = telescope.multiply_by_wavenumber(frequency);
            if (err) return err;

            // Get pointer to the station.
            oskar_StationModel* station = &(telescope.station[station_id]);

            // Start simulation.
            for (int j = 0; j < num_times; ++j)
            {
                // Start time for the visibility dump, in MJD(UTC).
                printf("--> Generating beam for snapshot (%i / %i).\n",
                        j+1, num_times);
                double t_dump = obs_start_mjd_utc + j * dt_dump;
                double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

                // Evaluate horizontal l,m,n for beam phase centre.
                double beam_l, beam_m, beam_n;
                err = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                        &beam_n, station, gast);
                if (err) return err;

                // Evaluate horizontal l,m,n coordinates.
                err = oskar_evaluate_source_horizontal_lmn(&l, &m, &n,
                        &RA, &Dec, station, gast);
                if (err) return err;

                // Evaluate the station beam.
                err = oskar_evaluate_station_beam(&beam_pattern, station,
                        beam_l, beam_m, &l, &m, &n, &weights, &curand_state);
                if (err) return err;

                // Copy beam pattern back to host memory.
                err = oskar_mem_copy(&beam_cpu, &beam_pattern);
                if (err) return err;

                // Convert from real/imaginary to power.
                int offset = (j + c * num_times) * num_pixels;
                if (type == OSKAR_SINGLE)
                {
                    float* img = (float*)data.data + offset;
                    float2* tc = (float2*)beam_cpu;
                    for (int i = 0; i < num_pixels; ++i)
                    {
                        float tx = tc[i].x;
                        float ty = tc[i].y;
                        img[i] = sqrt(tx * tx + ty * ty);
                    }
                }
                else if (type == OSKAR_DOUBLE)
                {
                    double* img = (double*)data.data + offset;
                    double2* tc = (double2*)beam_cpu;
                    for (int i = 0; i < num_pixels; ++i)
                    {
                        double tx = tc[i].x;
                        double ty = tc[i].y;
                        img[i] = sqrt(tx * tx + ty * ty);
                    }
                }
            }
        }
        printf("=== Simulation completed in %f sec.\n", timer.elapsed() / 1e3);
    }

    // Dump data to OSKAR image file if required.
    if (settings.beam_pattern.filename)
    {
        err = oskar_image_write(&data, settings.beam_pattern.filename);
        if (err) return err;
    }

#ifndef OSKAR_NO_FITS
    // FITS library available.
    if (settings.beam_pattern.fits_image)
    {
        oskar_fits_image_write(&data, settings.beam_pattern.fits_image);
    }
#endif

    // Delete telescope model.
    delete tel_cpu;
    cudaDeviceReset();

    oskar_settings_free(&settings);
    return OSKAR_SUCCESS;
}
