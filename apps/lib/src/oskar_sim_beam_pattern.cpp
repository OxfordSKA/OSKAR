/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <fits/oskar_fits_image_write.h>
#include <oskar_telescope.h>
#include <oskar_image_resize.h>
#include <oskar_image_write.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_ra_dec_to_rel_lmn.h>
#include <oskar_evaluate_image_lon_lat_grid.h>
#include <oskar_evaluate_source_horizontal_lmn.h>
#include <oskar_evaluate_station_beam_aperture_array.h>
#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_station_work.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_random_state.h>
#include <oskar_log.h>
#include <oskar_settings_free.h>

#include <cmath>
#include <QtCore/QTime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void oskar_set_up_beam_pattern(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status);

static void save_total_intensity(const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log, int* status);

extern "C"
int oskar_sim_beam_pattern(const char* settings_file, oskar_Log* log)
{
    int err = 0;
    const char* filename;
    QTime timer;

    // Load the settings file.
    oskar_Settings settings;
    oskar_log_section(log, "Loading settings file '%s'", settings_file);
    err = oskar_settings_load(&settings, log, settings_file);
    if (err) return err;
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_beam_pattern(log, &settings);

    // Check that a data file has been specified.
    if (!(settings.beam_pattern.oskar_image_voltage ||
            settings.beam_pattern.oskar_image_phase ||
            settings.beam_pattern.oskar_image_complex ||
            settings.beam_pattern.oskar_image_total_intensity ||
            settings.beam_pattern.fits_image_voltage ||
            settings.beam_pattern.fits_image_phase ||
            settings.beam_pattern.fits_image_total_intensity))
    {
        oskar_log_error(log, "No output file(s) specified.");
        return OSKAR_ERR_SETTINGS;
    }

    // Get time data.
    int num_times            = settings.obs.num_time_steps;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump           = settings.obs.dt_dump_days;

    // Get the telescope model.
    oskar_Telescope* tel = oskar_set_up_telescope(log, &settings, &err);
    if (err) return err;

    // Get the beam pattern settings.
    int station_id = settings.beam_pattern.station_id;
    int* image_size = settings.beam_pattern.size;
    int num_channels = settings.obs.num_channels;
    int num_pols = settings.telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    int num_pixels = image_size[0] * image_size[1];
    int num_pixels_total = num_pixels * num_times * num_channels * num_pols;
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4)
        beam_pattern_data_type |= OSKAR_MATRIX;

    // Check station ID is within range.
    if (station_id < 0 || station_id >= oskar_telescope_num_stations(tel))
        return OSKAR_ERR_OUT_OF_RANGE;

    // Set up beam pattern hyper-cubes.
    oskar_Image complex_cube(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    oskar_Image image_cube(type, OSKAR_LOCATION_CPU);
    oskar_set_up_beam_pattern(&complex_cube, &settings, num_pols, &err);
    oskar_set_up_beam_pattern(&image_cube, &settings, num_pols, &err);
    if (err) return err;

    // Temporary CPU memory, used to re-order polarisation data.
    oskar_Mem beam_cpu(beam_pattern_data_type, OSKAR_LOCATION_CPU, num_pixels);

    // All GPU memory used within these braces.
    {
        oskar_Mem RA, Dec, beam_pattern, l, m, n;
        oskar_StationWork* work;
        oskar_Mem *hor_x, *hor_y, *hor_z;
        const oskar_Station* station;
        double ra0, dec0;

        // Generate equatorial coordinates for beam pattern pixels.
        ra0 = oskar_telescope_ra0_rad(tel);
        dec0 = oskar_telescope_dec0_rad(tel);
        oskar_mem_init(&RA, type, OSKAR_LOCATION_GPU, num_pixels, 1, &err);
        oskar_mem_init(&Dec, type, OSKAR_LOCATION_GPU, num_pixels, 1, &err);
        oskar_evaluate_image_lon_lat_grid(&RA, &Dec, image_size[0],
                image_size[1], settings.beam_pattern.fov_deg[0] * (M_PI/180.0),
                settings.beam_pattern.fov_deg[1] * (M_PI/180.0), ra0, dec0,
                &err);

        // Initialise work array and GPU memory for a beam pattern.
        work = oskar_station_work_create(type, OSKAR_LOCATION_GPU, &err);
        hor_x = oskar_station_work_source_horizontal_x(work);
        hor_y = oskar_station_work_source_horizontal_y(work);
        hor_z = oskar_station_work_source_horizontal_z(work);
        oskar_mem_init(&beam_pattern, beam_pattern_data_type,
                OSKAR_LOCATION_GPU, num_pixels, 1, &err);
        oskar_mem_init(&l, type, OSKAR_LOCATION_GPU, 0, 1, &err);
        oskar_mem_init(&m, type, OSKAR_LOCATION_GPU, 0, 1, &err);
        oskar_mem_init(&n, type, OSKAR_LOCATION_GPU, 0, 1, &err);

        // Evaluate source relative l,m,n values if not an aperture array.
        station = oskar_telescope_station_const(tel, station_id);
        if (oskar_station_station_type(station) != OSKAR_STATION_TYPE_AA)
        {
            oskar_ra_dec_to_rel_lmn(num_pixels, &RA, &Dec, ra0, dec0,
                    &l, &m, &n, &err);
        }

        // Loop over channels.
        oskar_log_section(log, "Starting simulation...");
        timer.start();
        for (int c = 0; c < num_channels; ++c)
        {
            oskar_RandomState* random_state;
            oskar_Telescope* tel_gpu;
            double frequency, lon, lat;

            // Initialise local data structures.
            random_state = oskar_random_state_create(
                    oskar_telescope_max_station_size(tel),
                    oskar_telescope_random_seed(tel), 0, 0, &err);

            // Get the channel frequency.
            frequency = settings.obs.start_frequency_hz +
                    c * settings.obs.frequency_inc_hz;
            oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                    c + 1, num_channels, frequency / 1e6);

            // Copy the telescope model.
            tel_gpu = oskar_telescope_create_copy(tel,
                    OSKAR_LOCATION_GPU, &err);
            station = oskar_telescope_station_const(tel_gpu, station_id);
            lon = oskar_station_longitude_rad(station);
            lat = oskar_station_latitude_rad(station);

            // Loop over times.
            for (int t = 0; t < num_times; ++t)
            {
                double t_dump, gast, last;

                // Check error code.
                if (err) continue;

                // Start time for the data dump, in MJD(UTC).
                oskar_log_message(log, 1, "Snapshot %4d/%d",
                        t+1, num_times);
                t_dump = obs_start_mjd_utc + t * dt_dump;
                gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);
                last = gast + lon;

                // Evaluate horizontal x,y,z directions for source positions.
                oskar_evaluate_source_horizontal_lmn(num_pixels,
                        hor_x, hor_y, hor_z, &RA, &Dec, last, lat, &err);

                // Evaluate the station beam.
                if (oskar_station_station_type(station) ==
                        OSKAR_STATION_TYPE_AA)
                {
                    oskar_evaluate_station_beam_aperture_array(&beam_pattern,
                            station, num_pixels, hor_x, hor_y, hor_z, gast,
                            frequency, work, random_state, &err);
                }
                else if (oskar_station_station_type(station) ==
                        OSKAR_STATION_TYPE_GAUSSIAN_BEAM)
                {
                    oskar_evaluate_station_beam_gaussian(&beam_pattern,
                            num_pixels, &l, &m, hor_z,
                            oskar_station_gaussian_beam_fwhm_rad(station), &err);
                }
                else
                {
                    return OSKAR_ERR_SETTINGS_TELESCOPE;
                }

                // Save complex beam pattern data in the right order.
                // Cube has dimension order (from slowest to fastest):
                // Channel, Time, Polarisation, Declination, Right Ascension.
                int offset = (t + c * num_times) * num_pols * num_pixels;
                if (oskar_mem_is_scalar(&beam_pattern))
                {
                    oskar_mem_insert(&complex_cube.data, &beam_pattern,
                            offset, &err);
                }
                else
                {
                    // Copy beam pattern back to host memory for re-ordering.
                    oskar_mem_copy(&beam_cpu, &beam_pattern, &err);
                    if (err) continue;

                    // Re-order the polarisation data.
                    if (type == OSKAR_SINGLE)
                    {
                        float2* bp = (float2*)complex_cube.data.data + offset;
                        float4c* tc = (float4c*)beam_cpu.data;
                        for (int i = 0; i < num_pixels; ++i)
                        {
                            bp[i]                  = tc[i].a; // theta_X
                            bp[i +     num_pixels] = tc[i].b; // phi_X
                            bp[i + 2 * num_pixels] = tc[i].c; // theta_Y
                            bp[i + 3 * num_pixels] = tc[i].d; // phi_Y
                        }
                    }
                    else if (type == OSKAR_DOUBLE)
                    {
                        double2* bp = (double2*)complex_cube.data.data + offset;
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

            // Record GPU memory usage.
            oskar_cuda_mem_log(log, 1, 0);

            // Free memory.
            oskar_random_state_free(random_state, &err);
            oskar_telescope_free(tel_gpu, &err);
        } // Channel loop

        // Free memory.
        oskar_mem_free(&RA, &err);
        oskar_mem_free(&Dec, &err);
        oskar_mem_free(&beam_pattern, &err);
        oskar_mem_free(&l, &err);
        oskar_mem_free(&m, &err);
        oskar_mem_free(&n, &err);
        oskar_station_work_free(work, &err);
    } // GPU memory section
    oskar_log_section(log, "Simulation completed in %.3f sec.",
            timer.elapsed() / 1e3);

    oskar_telescope_free(tel, &err);

    // Write out complex data if required.
    filename = settings.beam_pattern.oskar_image_complex;
    if (filename && !err)
    {
        oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(&complex_cube, log, filename, 0, &err);
    }

    // Check error code.
    if (err) return err;

    // Write out power data if required.
    if (settings.beam_pattern.oskar_image_voltage ||
            settings.beam_pattern.fits_image_voltage)
    {
        // Convert complex values to power (amplitude of complex number).
        if (type == OSKAR_SINGLE)
        {
            float* image_data = oskar_mem_float(&image_cube.data, &err);
            float2* complex_data = oskar_mem_float2(&complex_cube.data, &err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                float x, y;
                x = complex_data[i].x;
                y = complex_data[i].y;
                image_data[i] = sqrt(x*x + y*y);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data = oskar_mem_double(&image_cube.data, &err);
            double2* complex_data = oskar_mem_double2(&complex_cube.data, &err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                double x, y;
                x = complex_data[i].x;
                y = complex_data[i].y;
                image_data[i] = sqrt(x*x + y*y);
            }
        }

        // Write OSKAR image.
        filename = settings.beam_pattern.oskar_image_voltage;
        if (filename && !err)
        {
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(&image_cube, log, filename, 0, &err);
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        filename = settings.beam_pattern.fits_image_voltage;
        if (filename && !err)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(&image_cube, log, filename, &err);
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
            float* image_data = oskar_mem_float(&image_cube.data, &err);
            float2* complex_data = oskar_mem_float2(&complex_cube.data, &err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data = oskar_mem_double(&image_cube.data, &err);
            double2* complex_data = oskar_mem_double2(&complex_cube.data, &err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }

        // Write OSKAR image.
        filename = settings.beam_pattern.oskar_image_phase;
        if (filename && !err)
        {
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(&image_cube, log, filename, 0, &err);
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        filename = settings.beam_pattern.fits_image_phase;
        if (filename && !err)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(&image_cube, log, filename, &err);
        }
#endif
    }

    // Save the beam pattern as a total intensity pattern (if requested).
    save_total_intensity(complex_cube, settings, type, log, &err);

    cudaDeviceReset();
    return err;
}

static void oskar_set_up_beam_pattern(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status)
{
    int num_channels = settings->obs.num_channels;
    int num_times = settings->obs.num_time_steps;
    const int* image_size = settings->beam_pattern.size;

    /* Resize image cube. */
    oskar_image_resize(image, image_size[0], image_size[1], num_pols, num_times,
            num_channels, status);

    /* Set beam pattern meta-data. */
    image->image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    image->centre_ra_deg      = settings->obs.ra0_rad[0] * 180.0 / M_PI;
    image->centre_dec_deg     = settings->obs.dec0_rad[0] * 180.0 / M_PI;
    image->fov_ra_deg         = settings->beam_pattern.fov_deg[0];
    image->fov_dec_deg        = settings->beam_pattern.fov_deg[1];
    image->freq_start_hz      = settings->obs.start_frequency_hz;
    image->freq_inc_hz        = settings->obs.frequency_inc_hz;
    image->time_inc_sec       = settings->obs.dt_dump_days * 86400.0;
    image->time_start_mjd_utc = settings->obs.start_mjd_utc;
    oskar_mem_copy(&image->settings_path, &settings->settings_path, status);
}


// complex_cube == voltage image cube
static void save_total_intensity(const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log, int* status)
{
    const char* filename;
    if (*status) return;

    // Return if a total intensity beam pattern has not been specified.
    if (!(settings.beam_pattern.oskar_image_total_intensity ||
            settings.beam_pattern.fits_image_total_intensity))
        return;

    // Dimensions of input beam pattern image to be converted to total intensity.
    int num_channels = complex_cube.num_channels;
    int num_times = complex_cube.num_times;
    int num_pols = complex_cube.num_pols;
    int num_pixels = complex_cube.width * complex_cube.height;

    // Allocate total intensity image cube to write into.
    oskar_Image image_cube_I(type, OSKAR_LOCATION_CPU);
    oskar_set_up_beam_pattern(&image_cube_I, &settings, 1, status);

    // For polarised beams Stokes I is 0.5 * (XX + YY)
    // For scalar beams total intensity is voltage squared.
    double factor = (num_pols == 4) ? 0.5 : 1.0;

    if (type == OSKAR_SINGLE)
    {
        float* image_data = oskar_mem_float(&image_cube_I.data, status);
        const float2* complex_data =
                oskar_mem_float2_const(&complex_cube.data, status);
        for (int c = 0, idx = 0, islice = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t, ++islice)
            {
                float* image_plane = &(image_data[islice * num_pixels]);
                for (int p = 0; p < num_pols; ++p)
                {
                    for (int i = 0; i < num_pixels; ++i, ++idx)
                    {
                        float xx = complex_data[idx].x * complex_data[idx].x;
                        float yy = complex_data[idx].y * complex_data[idx].y;
                        image_plane[i] += factor * (xx + yy);
                    }
                }
            }
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        double* image_data = oskar_mem_double(&image_cube_I.data, status);
        const double2* complex_data =
                oskar_mem_double2_const(&complex_cube.data, status);
        for (int c = 0, idx = 0, islice = 0; c < num_channels; ++c)
        {
            for (int t = 0; t < num_times; ++t, ++islice)
            {
                double* image_plane = &(image_data[islice * num_pixels]);
                for (int p = 0; p < num_pols; ++p)
                {
                    for (int i = 0; i < num_pixels; ++i, ++idx)
                    {
                        double xx = complex_data[idx].x * complex_data[idx].x;
                        double yy = complex_data[idx].y * complex_data[idx].y;
                        image_plane[i] += factor * (xx + yy);
                    }
                }
            }
        }
    }

    // Write OSKAR image.
    filename = settings.beam_pattern.oskar_image_total_intensity;
    if (filename && !*status)
    {
        oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(&image_cube_I, log, filename, 0, status);
    }
#ifndef OSKAR_NO_FITS
    // Write FITS image.
    filename = settings.beam_pattern.fits_image_total_intensity;
    if (filename && !*status)
    {
        oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
        oskar_fits_image_write(&image_cube_I, log, filename, status);
    }
#endif
}

