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
#include "fits/oskar_fits_write.h"
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
    if (!settings.image.filename)
    {
        fprintf(stderr, "ERROR: No image file specified.\n");
        return OSKAR_ERR_SETTINGS;
    }

    // Get the sky model and telescope model and copy both to GPU (slow step).
    oskar_TelescopeModel* tel_cpu, *tel_gpu;
    tel_cpu = oskar_set_up_telescope(&settings);
    tel_gpu = new oskar_TelescopeModel(tel_cpu, OSKAR_LOCATION_GPU);

    // Get the image settings.
    int image_size = settings.image.size;
    int num_channels = settings.obs.num_channels;
    int num_pixels = image_size * image_size;
    double fov_deg = settings.image.fov_deg;
    double lm_max = sin((fov_deg / 2.0) * M_PI / 180.0);
    double ra0 = settings.obs.ra0_rad;
    double dec0 = settings.obs.dec0_rad;

    // Get time data.
    int num_vis_dumps        = times->num_vis_dumps;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;

    // Allocate CPU memory big enough for data hyper-cube.
    int num_elements = num_pixels * num_vis_dumps * num_channels;
    oskar_Mem data(type, OSKAR_LOCATION_CPU, num_elements);
    oskar_Mem image(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem beam_cpu(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU, num_pixels);

    // Generate l,m grid and equatorial coordinates for beam pattern pixels.
    oskar_Mem l_cpu(type, OSKAR_LOCATION_CPU, image_size);
    oskar_Mem m_cpu(type, OSKAR_LOCATION_CPU, image_size);
    oskar_Mem grid_l(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem grid_m(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem RA_cpu(type, OSKAR_LOCATION_CPU, num_pixels);
    oskar_Mem Dec_cpu(type, OSKAR_LOCATION_CPU, num_pixels);

    /*
     * Note that FITS images conventionally have the LARGEST value of
     * RA (=longitude) and the SMALLEST value of DEC (=latitude) at the
     * lowest memory address, so therefore the grid l-values must start off
     * positive and go negative, while the grid m-values start off negative
     * and go positive.
     */
    if (type == OSKAR_SINGLE)
    {
        oskar_linspace_f(l_cpu, lm_max, -lm_max, image_size); // FITS convention.
        oskar_linspace_f(m_cpu, -lm_max, lm_max, image_size);

        // Slowest varying is m, fastest varying is l.
        for (int j = 0, p = 0; j < image_size; ++j)
        {
            for (int i = 0; i < image_size; ++i, ++p)
            {
                ((float*)grid_l)[p] = ((float*)l_cpu)[i];
                ((float*)grid_m)[p] = ((float*)m_cpu)[j];
            }
        }
        oskar_sph_from_lm_f(num_pixels, ra0, dec0,
                grid_l, grid_m, RA_cpu, Dec_cpu);
    }
    else if (type == OSKAR_DOUBLE)
    {
        oskar_linspace_d(l_cpu, lm_max, -lm_max, image_size); // FITS convention.
        oskar_linspace_d(m_cpu, -lm_max, lm_max, image_size);

        // Slowest varying is m, fastest varying is l.
        for (int j = 0, p = 0; j < image_size; ++j)
        {
            for (int i = 0; i < image_size; ++i, ++p)
            {
                ((double*)grid_l)[p] = ((double*)l_cpu)[i];
                ((double*)grid_m)[p] = ((double*)m_cpu)[j];
            }
        }
        oskar_sph_from_lm_d(num_pixels, ra0, dec0,
                grid_l, grid_m, RA_cpu, Dec_cpu);
    }

    // All GPU memory used within these braces.
    {
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
            oskar_Device_curand_state curand_state(tel_cpu->max_station_size);
            int seed = 0; // TODO get this from the settings file....
//            curand_state.init(seed);

            // Get the channel frequency.
            printf("\n--> Simulating channel (%d / %d).\n", c+1, num_channels);
            double frequency = settings.obs.start_frequency_hz +
                    c * settings.obs.frequency_inc_hz;

            // Copy the telescope model and scale coordinates to wavenumbers.
            oskar_TelescopeModel telescope(tel_gpu, OSKAR_LOCATION_GPU);
            err = telescope.multiply_by_wavenumber(frequency); // TODO CHECK.
            if (err) return err;

            // Get pointer to the station.
            // FIXME Currently station 0: Determine this from the settings file?
            oskar_StationModel* station = &(telescope.station[0]);

            // Start simulation.
            for (int j = 0; j < num_vis_dumps; ++j)
            {
                // Start time for the visibility dump, in MJD(UTC).
                printf("--> Generating beam for snapshot (%i / %i).\n",
                        j+1, num_vis_dumps);
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
                if (type == OSKAR_SINGLE)
                {
                    float* img = (float*)image;
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
                    double* img = (double*)image;
                    double2* tc = (double2*)beam_cpu;
                    for (int i = 0; i < num_pixels; ++i)
                    {
                        double tx = tc[i].x;
                        double ty = tc[i].y;
                        img[i] = sqrt(tx * tx + ty * ty);
                    }
                }

                // Save to global data store.
                int offset = (j + c * num_vis_dumps) * num_pixels;
                err = oskar_mem_insert(&data, &image, offset);
                if (err) return err;
            }
        }
        printf("=== Simulation completed in %f sec.\n", timer.elapsed() / 1e3);
    }

    // Dump data to file.
#ifdef OSKAR_NO_FITS
    // No FITS library available.
    // Write OSKAR binary file.
#else
    // FITS file OK.
    long naxes[4];
    double crval[4], crpix[4], cdelt[4];
    double crota[] = {0.0, 0.0, 0.0, 0.0};

    // Axis types.
    const char* ctype[] = {
            "RA---SIN",
            "DEC--SIN",
            "UTC",
            "FREQ"
    };

    // Axis comments.
    const char* ctype_comment[] = {
            "Right Ascension",
            "Declination",
            "Time",
            "Frequency"
    };

    // Axis dimensions.
    naxes[0] = image_size; // width
    naxes[1] = image_size; // height
    naxes[2] = num_vis_dumps;
    naxes[3] = num_channels;

    // Reference values.
    crval[0] = ra0 * 180.0 / M_PI;
    crval[1] = dec0 * 180.0 / M_PI;
    crval[2] = obs_start_mjd_utc;
    crval[3] = settings.obs.start_frequency_hz;

    // Deltas.
    cdelt[0] = -(fov_deg / image_size); // DELTA_RA
    cdelt[1] = (fov_deg / image_size); // DELTA_DEC
    cdelt[2] = dt_dump; // DELTA_TIME
    cdelt[3] = settings.obs.frequency_inc_hz; // DELTA_CHANNEL

    // Reference pixels.
    crpix[0] = (image_size + 1) / 2.0;
    crpix[1] = (image_size + 1) / 2.0;
    crpix[2] = 1.0;
    crpix[3] = 1.0;

    // Write multi-dimensional image data.
    oskar_fits_write(settings.image.filename, data.type(), 4, naxes,
            data.data, ctype, ctype_comment, crval, cdelt, crpix, crota);
#endif

    // Delete data structures.
    delete tel_gpu;
    delete tel_cpu;
    cudaDeviceReset();

    oskar_settings_free(&settings);
    return OSKAR_SUCCESS;
}
