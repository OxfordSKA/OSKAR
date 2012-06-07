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
#include "utility/oskar_Work.h"

#include <cmath>
#include <QtCore/QTime>

extern "C"
int oskar_sim_beam_pattern(const char* settings_file, oskar_Log* log)
{
    int err;
    QTime timer;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    err = oskar_settings_load(&settings, log, settings_file);
    if (err) return err;
    const oskar_SettingsTime* times = &settings.obs.time;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Log the relevant settings.
    log->keep_file = settings.sim.keep_log_file;
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_beam_pattern(log, &settings);

    // Check that a data file has been specified.
    if (!(settings.beam_pattern.oskar_image_filename ||
            settings.beam_pattern.fits_image_filename ||
            settings.beam_pattern.oskar_voltage_pattern_binary))
    {
        oskar_log_error(log, "No output file(s) specified.");
        return OSKAR_ERR_SETTINGS;
    }

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
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4)
        beam_pattern_data_type |= OSKAR_MATRIX;
    double fov = settings.beam_pattern.fov_deg * M_PI / 180;
    double ra0 = settings.obs.ra0_rad;
    double dec0 = settings.obs.dec0_rad;

    // Check station ID is within range.
    if (station_id < 0 || station_id >= tel_cpu.num_stations)
        return OSKAR_ERR_OUT_OF_RANGE;

    // Get time data.
    int num_times            = times->num_time_steps;
    double obs_start_mjd_utc = times->obs_start_mjd_utc;
    double dt_dump           = times->dt_dump_days;

    // Declare image hyper-cube.
    oskar_Image image_cube(type, OSKAR_LOCATION_CPU);
    err = oskar_image_resize(&image_cube, image_size, image_size, num_pols,
            num_times, num_channels);
    if (err) return err;

    // Set image meta-data.
    image_cube.image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    image_cube.centre_ra_deg      = settings.obs.ra0_rad * 180.0 / M_PI;
    image_cube.centre_dec_deg     = settings.obs.dec0_rad * 180.0 / M_PI;
    image_cube.fov_ra_deg         = settings.beam_pattern.fov_deg;
    image_cube.fov_dec_deg        = settings.beam_pattern.fov_deg;
    image_cube.freq_start_hz      = settings.obs.start_frequency_hz;
    image_cube.freq_inc_hz        = settings.obs.frequency_inc_hz;
    image_cube.time_inc_sec       = settings.obs.time.dt_dump_days * 86400.0;
    image_cube.time_start_mjd_utc = settings.obs.time.obs_start_mjd_utc;
    err = oskar_mem_copy(&image_cube.settings_path, &settings.settings_path);
    if (err) return err;


    // Declare image hyper-cube for voltage pattern.
    // NOTE this, along with the image cube could allocate unacceptable
    // amounts of memory in some cases.
    // --- need a mode to write a slice at a time?
    oskar_Image voltage_pattern_cube(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    if (settings.beam_pattern.oskar_voltage_pattern_binary)
    {
        err = oskar_image_resize(&voltage_pattern_cube, image_size, image_size,
                num_pols, num_times, num_channels);
        if (err) return err;

        // Set voltage pattern meta-data.
        voltage_pattern_cube.image_type         = (num_pols == 1) ?
                OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
        voltage_pattern_cube.centre_ra_deg      = settings.obs.ra0_rad * 180.0 / M_PI;
        voltage_pattern_cube.centre_dec_deg     = settings.obs.dec0_rad * 180.0 / M_PI;
        voltage_pattern_cube.fov_ra_deg         = settings.beam_pattern.fov_deg;
        voltage_pattern_cube.fov_dec_deg        = settings.beam_pattern.fov_deg;
        voltage_pattern_cube.freq_start_hz      = settings.obs.start_frequency_hz;
        voltage_pattern_cube.freq_inc_hz        = settings.obs.frequency_inc_hz;
        voltage_pattern_cube.time_inc_sec       = settings.obs.time.dt_dump_days * 86400.0;
        voltage_pattern_cube.time_start_mjd_utc = settings.obs.time.obs_start_mjd_utc;
        err = oskar_mem_copy(&voltage_pattern_cube.settings_path, &settings.settings_path);
        if (err) return err;
    }

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

        // Copy RA and Dec to GPU and allocate arrays for direction cosines.
        oskar_Mem RA(&RA_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem Dec(&Dec_cpu, OSKAR_LOCATION_GPU);
        oskar_Mem l(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem m(type, OSKAR_LOCATION_GPU, num_pixels);
        oskar_Mem n(type, OSKAR_LOCATION_GPU, num_pixels);

        // Allocate weights work array and GPU memory for a beam pattern.
        oskar_Mem weights(type | OSKAR_COMPLEX, OSKAR_LOCATION_GPU);
        oskar_Mem beam_pattern(beam_pattern_data_type, OSKAR_LOCATION_GPU,
                num_pixels);
        oskar_Work work(type, OSKAR_LOCATION_GPU);

        // Loop over channels.
        oskar_log_section(log, "Starting simulation...");
        timer.start();
        for (int c = 0; c < num_channels; ++c)
        {
            // Initialise the random number generator.
            oskar_Device_curand_state curand_state(tel_gpu.max_station_size);
            curand_state.init(tel_gpu.seed_time_variable_errors);

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
                err = oskar_evaluate_source_horizontal_lmn(num_pixels,
                        &l, &m, &n, &RA, &Dec, station, gast);
                if (err) return err;

                // Evaluate the station beam.
                err = oskar_evaluate_station_beam(&beam_pattern, station,
                        beam_l, beam_m, beam_n, &l, &m, &n, &work,
                        &curand_state);
                if (err) return err;

                // Copy beam pattern back to host memory.
                err = oskar_mem_copy(&beam_cpu, &beam_pattern);
                if (err) return err;

                // Convert from complex to power or phase.
                // Image cube has dimension order (from slowest to fastest):
                // Channel, Time, Polarisation, Declination, Right Ascension.
                int offset = (t + c * num_times) * num_pols * num_pixels;
                if (type == OSKAR_SINGLE)
                {
                    float* img = (float*)image_cube.data + offset;
                    float2* bp = NULL;
                    if (settings.beam_pattern.oskar_voltage_pattern_binary)
                    {
                        bp = &((float2*)voltage_pattern_cube.data)[offset];
                    }
                    float tx, ty;
                    if (oskar_mem_is_scalar(beam_pattern_data_type))
                    {
                        float2* tc = (float2*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            tx = tc[i].x;
                            ty = tc[i].y;
                            img[i] = sqrt(tx * tx + ty * ty);
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i].x = tx;
                                bp[i].y = ty;
                            }
                        }
                    }
                    else
                    {
                        float4c* tc = (float4c*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            tx = tc[i].a.x;
                            ty = tc[i].a.y;
                            img[i] = sqrt(tx * tx + ty * ty); // theta_X
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i].x = tx;
                                bp[i].y = ty;
                            }
                            tx = tc[i].b.x;
                            ty = tc[i].b.y;
                            img[i + num_pixels] = sqrt(tx * tx + ty * ty); // phi_X
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + num_pixels].x = tx;
                                bp[i + num_pixels].y = ty;
                            }
                            tx = tc[i].c.x;
                            ty = tc[i].c.y;
                            img[i + 2 * num_pixels] = sqrt(tx * tx + ty * ty); // theta_Y
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + 2 * num_pixels].x = tx;
                                bp[i + 2 * num_pixels].y = ty;
                            }
                            tx = tc[i].d.x;
                            ty = tc[i].d.y;
                            img[i + 3 * num_pixels] = sqrt(tx * tx + ty * ty); // phi_Y
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + 3 * num_pixels].x = tx;
                                bp[i + 3 * num_pixels].y = ty;
                            }
                        }
                    }
                }
                else if (type == OSKAR_DOUBLE)
                {
                    double* img = (double*)image_cube.data + offset;
                    double2* bp = NULL;
                    if (settings.beam_pattern.oskar_voltage_pattern_binary)
                    {
                        bp = &((double2*)voltage_pattern_cube.data)[offset];
                    }
                    double tx, ty;
                    if (oskar_mem_is_scalar(beam_pattern_data_type))
                    {
                        double2* tc = (double2*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            tx = tc[i].x;
                            ty = tc[i].y;
                            img[i] = sqrt(tx * tx + ty * ty);
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i].x = tx;
                                bp[i].y = ty;
                            }
                        }
                    }
                    else
                    {
                        double4c* tc = (double4c*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            tx = tc[i].a.x;
                            ty = tc[i].a.y;
                            img[i] = sqrt(tx * tx + ty * ty); // theta_X
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i].x = tx;
                                bp[i].y = ty;
                            }
                            tx = tc[i].b.x;
                            ty = tc[i].b.y;
                            img[i + num_pixels] = sqrt(tx * tx + ty * ty); // phi_X
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + num_pixels].x = tx;
                                bp[i + num_pixels].y = ty;
                            }
                            tx = tc[i].c.x;
                            ty = tc[i].c.y;
                            img[i + 2 * num_pixels] = sqrt(tx * tx + ty * ty); // theta_Y
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + 2 * num_pixels].x = tx;
                                bp[i + 2 * num_pixels].y = ty;
                            }
                            tx = tc[i].d.x;
                            ty = tc[i].d.y;
                            img[i + 3 * num_pixels] = sqrt(tx * tx + ty * ty); // phi_Y
                            if (settings.beam_pattern.oskar_voltage_pattern_binary)
                            {
                                bp[i + 3 * num_pixels].x = tx;
                                bp[i + 3 * num_pixels].y = ty;
                            }

                        }
                    }
                }

            } // Time loop
        } // Channel loop
    } // GPU memory section
    oskar_log_section(log, "Simulation completed in %.3f sec.",
            timer.elapsed() / 1e3);

    // Dump data to OSKAR image file if required.
    if (settings.beam_pattern.oskar_image_filename)
    {
        err = oskar_image_write(&image_cube, log,
                settings.beam_pattern.oskar_image_filename, 0);
        if (err) return err;
    }

#ifndef OSKAR_NO_FITS
    // FITS library available.
    if (settings.beam_pattern.fits_image_filename)
    {
        err = oskar_fits_image_write(&image_cube, log,
                settings.beam_pattern.fits_image_filename);
        if (err) return err;
    }
#endif

    // Write voltage pattern image file (if required).
    if (settings.beam_pattern.oskar_voltage_pattern_binary)
    {
        err = oskar_image_write(&voltage_pattern_cube, log,
                settings.beam_pattern.oskar_voltage_pattern_binary, 0);
        if (err) return err;
    }

    cudaDeviceReset();
    return OSKAR_SUCCESS;
}
