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
#include "apps/lib/oskar_settings_load.h"
#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_sim_beam_pattern.h"
#include "fits/oskar_fits_image_write.h"
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
#include "utility/oskar_log_error.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_settings.h"
#include "utility/oskar_Log.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_insert.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_settings_free.h"
#include "utility/oskar_vector_types.h"

#include <cmath>
#include <QtCore/QTime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
int oskar_sim_beam_pattern(const char* settings_file, oskar_Log* log)
{
    int err = 0;
    QTime timer;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    err = oskar_settings_load(&settings, log, settings_file);
    if (err) return err;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Log the relevant settings.
    log->keep_file = settings.sim.keep_log_file;
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_beam_pattern(log, &settings);

    // Check that a data file has been specified.
    if (!(settings.beam_pattern.oskar_image_power ||
            settings.beam_pattern.oskar_image_phase ||
            settings.beam_pattern.oskar_image_complex ||
            settings.beam_pattern.fits_image_power ||
            settings.beam_pattern.fits_image_phase))
    {
        oskar_log_error(log, "No output file(s) specified.");
        return OSKAR_ERR_SETTINGS;
    }

    // Get time data.
    int num_times            = settings.obs.num_time_steps;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump           = settings.obs.dt_dump_days;

    // Get the telescope model.
    oskar_TelescopeModel tel_cpu;
    err = oskar_set_up_telescope(&tel_cpu, log, &settings);
    if (err) return OSKAR_ERR_SETUP_FAIL;

    // Get the beam pattern settings.
    int station_id = settings.beam_pattern.station_id;
    int image_size = settings.beam_pattern.size;
    int num_channels = settings.obs.num_channels;
    int num_pols = settings.telescope.station.use_polarised_elements ? 4 : 1;
    int num_pixels = image_size * image_size;
    int num_pixels_total = num_pixels * num_times * num_channels * num_pols;
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4)
        beam_pattern_data_type |= OSKAR_MATRIX;
    double fov = settings.beam_pattern.fov_deg * M_PI / 180;
    double ra0 = settings.obs.ra0_rad;
    double dec0 = settings.obs.dec0_rad;

    // Check station ID is within range.
    if (station_id < 0 || station_id >= tel_cpu.num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;

    // Declare image hyper-cube for complex voltage pattern.
    oskar_Image complex_cube(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    err = oskar_image_resize(&complex_cube, image_size, image_size,
            num_pols, num_times, num_channels);
    if (err) return err;

    // Set complex voltage pattern meta-data.
    complex_cube.image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    complex_cube.centre_ra_deg      = settings.obs.ra0_rad * 180.0 / M_PI;
    complex_cube.centre_dec_deg     = settings.obs.dec0_rad * 180.0 / M_PI;
    complex_cube.fov_ra_deg         = settings.beam_pattern.fov_deg;
    complex_cube.fov_dec_deg        = settings.beam_pattern.fov_deg;
    complex_cube.freq_start_hz      = settings.obs.start_frequency_hz;
    complex_cube.freq_inc_hz        = settings.obs.frequency_inc_hz;
    complex_cube.time_inc_sec       = settings.obs.dt_dump_days * 86400.0;
    complex_cube.time_start_mjd_utc = settings.obs.start_mjd_utc;
    oskar_mem_copy(&complex_cube.settings_path, &settings.settings_path, &err);
    if (err) return err;

    // Declare image hyper-cube.
    oskar_Image image_cube(type, OSKAR_LOCATION_CPU);
    err = oskar_image_resize(&image_cube, image_size, image_size, num_pols,
            num_times, num_channels);
    if (err) return err;

    // Set image meta-data.
    image_cube.image_type         = complex_cube.image_type;
    image_cube.centre_ra_deg      = complex_cube.centre_ra_deg;
    image_cube.centre_dec_deg     = complex_cube.centre_dec_deg;
    image_cube.fov_ra_deg         = complex_cube.fov_ra_deg;
    image_cube.fov_dec_deg        = complex_cube.fov_dec_deg;
    image_cube.freq_start_hz      = complex_cube.freq_start_hz;
    image_cube.freq_inc_hz        = complex_cube.freq_inc_hz;
    image_cube.time_inc_sec       = complex_cube.time_inc_sec;
    image_cube.time_start_mjd_utc = complex_cube.time_start_mjd_utc;
    oskar_mem_copy(&image_cube.settings_path, &settings.settings_path, &err);
    if (err) return err;

    // Temporary CPU memory.
    oskar_Mem beam_cpu(beam_pattern_data_type, OSKAR_LOCATION_CPU, num_pixels);

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
        oskar_TelescopeModel tel_gpu(&tel_cpu, OSKAR_LOCATION_GPU);

        // Copy RA and Dec to GPU.
        oskar_Mem RA(&RA_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem Dec(&Dec_cpu, OSKAR_LOCATION_GPU);

        // Declare work array and GPU memory for a beam pattern.
        oskar_WorkStationBeam work(type, OSKAR_LOCATION_GPU);
        oskar_Mem beam_pattern(beam_pattern_data_type, OSKAR_LOCATION_GPU,
                num_pixels);

        // Loop over channels.
        oskar_log_section(log, "Starting simulation...");
        timer.start();
        for (int c = 0; c < num_channels; ++c)
        {
            // Initialise the random number generator.
            oskar_Device_curand_state curand_state(tel_gpu.max_station_size);
            curand_state.init(tel_gpu.seed_time_variable_station_element_errors);

            // Get the channel frequency.
            double frequency = settings.obs.start_frequency_hz +
                    c * settings.obs.frequency_inc_hz;
            oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                    c + 1, num_channels, frequency / 1e6);

            // Copy the telescope model and scale coordinates to radians.
            oskar_TelescopeModel telescope(&tel_gpu, OSKAR_LOCATION_GPU);
            err = telescope.multiply_by_wavenumber(frequency);
            if (err) return err;

            // Get pointer to the station.
            oskar_StationModel* station = &(telescope.station[station_id]);

            // Start simulation.
            for (int t = 0; t < num_times; ++t)
            {
                // Start time for the data dump, in MJD(UTC).
                oskar_log_message(log, 1, "Snapshot %4d/%d",
                        t+1, num_times);
                double t_dump = obs_start_mjd_utc + t * dt_dump;
                double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

                // Evaluate horizontal l,m,n for beam phase centre.
                double beam_l, beam_m, beam_n;
                err = oskar_evaluate_beam_horizontal_lmn(&beam_l, &beam_m,
                        &beam_n, station, gast);
                if (err) return err;

                // Evaluate horizontal l,m,n coordinates.
                oskar_evaluate_source_horizontal_lmn(num_pixels,
                        &work.hor_l, &work.hor_m, &work.hor_n, &RA, &Dec,
                        station, gast, &err);
                if (err) return err;

                // Evaluate the station beam.
                oskar_evaluate_station_beam(&beam_pattern, station,
                        beam_l, beam_m, beam_n, num_pixels, HORIZONTAL_XYZ,
                        &work.hor_l, &work.hor_m, &work.hor_n, &work.hor_n,
                        &work, &curand_state, &err);
                if (err) return err;

                // Copy beam pattern back to host memory.
                oskar_mem_copy(&beam_cpu, &beam_pattern, &err);
                if (err) return err;

                // Save complex beam pattern data in the right order.
                // Cube has dimension order (from slowest to fastest):
                // Channel, Time, Polarisation, Declination, Right Ascension.
                int offset = (t + c * num_times) * num_pols * num_pixels;
                if (type == OSKAR_SINGLE)
                {
                    float2* bp = (float2*)complex_cube.data + offset;
                    if (oskar_mem_is_scalar(beam_pattern_data_type))
                    {
                        float2* tc = (float2*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                            bp[i] = tc[i];
                    }
                    else
                    {
                        float4c* tc = (float4c*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            bp[i]                  = tc[i].a; // theta_X
                            bp[i +     num_pixels] = tc[i].b; // phi_X
                            bp[i + 2 * num_pixels] = tc[i].c; // theta_Y
                            bp[i + 3 * num_pixels] = tc[i].d; // phi_Y
                        }
                    }
                }
                else if (type == OSKAR_DOUBLE)
                {
                    double2* bp = (double2*)complex_cube.data + offset;
                    if (oskar_mem_is_scalar(beam_pattern_data_type))
                    {
                        double2* tc = (double2*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                            bp[i] = tc[i];
                    }
                    else
                    {
                        double4c* tc = (double4c*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            bp[i]                  = tc[i].a; // theta_X
                            bp[i +     num_pixels] = tc[i].b; // phi_X
                            bp[i + 2 * num_pixels] = tc[i].c; // theta_Y
                            bp[i + 3 * num_pixels] = tc[i].d; // phi_Y
                        }
                    }
                }

            } // Time loop
        } // Channel loop
    } // GPU memory section
    oskar_log_section(log, "Simulation completed in %.3f sec.",
            timer.elapsed() / 1e3);

    // Write out complex data if required.
    if (settings.beam_pattern.oskar_image_complex)
    {
        err = oskar_image_write(&complex_cube, log,
                settings.beam_pattern.oskar_image_complex, 0);
        if (err) return err;
    }

    // Write out power data if required.
    if (settings.beam_pattern.oskar_image_power ||
            settings.beam_pattern.fits_image_power)
    {
        // Convert complex values to power (amplitude of complex number).
        if (type == OSKAR_SINGLE)
        {
            float* image_data = (float*)image_cube.data;
            float2* complex_data = (float2*)complex_cube.data;
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = sqrt(complex_data[i].x*complex_data[i].x +
                        complex_data[i].y*complex_data[i].y);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data = (double*)image_cube.data;
            double2* complex_data = (double2*)complex_cube.data;
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = sqrt(complex_data[i].x*complex_data[i].x +
                        complex_data[i].y*complex_data[i].y);
            }
        }

        // Write OSKAR image
        if (settings.beam_pattern.oskar_image_power)
        {
            err = oskar_image_write(&image_cube, log,
                    settings.beam_pattern.oskar_image_power, 0);
            if (err) return err;
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        if (settings.beam_pattern.fits_image_power)
        {
            err = oskar_fits_image_write(&image_cube, log,
                    settings.beam_pattern.fits_image_power);
            if (err) return err;
        }
#endif
    }

    // Write out phase data if required.
    if (settings.beam_pattern.oskar_image_phase ||
            settings.beam_pattern.fits_image_phase)
    {
        // Convert complex values to phase.
        if (type == OSKAR_SINGLE)
        {
            float* image_data = (float*)image_cube.data;
            float2* complex_data = (float2*)complex_cube.data;
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data = (double*)image_cube.data;
            double2* complex_data = (double2*)complex_cube.data;
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }

        // Write OSKAR image
        if (settings.beam_pattern.oskar_image_phase)
        {
            err = oskar_image_write(&image_cube, log,
                    settings.beam_pattern.oskar_image_phase, 0);
            if (err) return err;
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        if (settings.beam_pattern.fits_image_phase)
        {
            err = oskar_fits_image_write(&image_cube, log,
                    settings.beam_pattern.fits_image_phase);
            if (err) return err;
        }
#endif
    }

    cudaDeviceReset();
    return OSKAR_SUCCESS;
}
