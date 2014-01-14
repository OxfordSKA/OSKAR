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

#include <apps/lib/oskar_sim_beam_pattern.h>
#include <apps/lib/oskar_settings_load.h>
#include <apps/lib/oskar_set_up_telescope.h>

#include <apps/lib/oskar_beam_pattern_generate_coordinates.h>
#include <oskar_evaluate_station_beam_pattern.h>
#include <apps/lib/oskar_beam_pattern_write.h>

#include <oskar_telescope.h>
#include <oskar_image_resize.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_station_work.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_random_state.h>
#include <oskar_log.h>

#include <oskar_settings_free.h>

#include <QtCore/QTime>

#include <cmath>
#include <cstdio>      // for remove()

// ============================================================================
static void load_settings_(oskar_Settings* settings, const char* filename,
        oskar_Log* log, int* status);
static void simulate_beam_pattern_cube_(oskar_Image* beam_pattern_cube,
        oskar_Settings* settings, int type, oskar_Telescope* tel,
        oskar_Log* log, int* status);
static void add_beam_pattern_to_complex_cube_(oskar_Image* beam_pattern_cube,
        oskar_Mem* beam_temp, const oskar_Mem* beam_pattern, int num_times,
        int num_pols, int num_pixels, int t, int c, int* status);
static void init_beam_pattern_cube_(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status);
// ============================================================================

extern "C"
void oskar_sim_beam_pattern(const char* settings_file, oskar_Log* log, int* status)
{
    // Load settings and telescope model.
    oskar_Settings settings;
    load_settings_(&settings, settings_file, log, status);
    oskar_Telescope* tel = oskar_set_up_telescope(log, &settings, status);
    if (*status)
    {
        if (tel) oskar_telescope_free(tel, status);
        return;
    }

    // Compute the beam pattern cube and write to file.
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Image beam_pattern(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    QTime timer;
    timer.start();
    simulate_beam_pattern_cube_(&beam_pattern, &settings, type, tel, log, status);
    double elapsed = timer.elapsed()/1.0e3;
    oskar_log_section(log, "Simulation completed in %.3f sec.", elapsed);
    oskar_beam_pattern_write(&beam_pattern, &settings, type, log, status);

    // Free memory.
    oskar_telescope_free(tel, status);

    cudaDeviceReset();
}

static void load_settings_(oskar_Settings* settings, const char* filename,
        oskar_Log* log, int* status)
{
    if (!status || *status != OSKAR_SUCCESS)
        return;
    if (!settings || !filename) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    oskar_log_section(log, "Loading settings file '%s'", filename);
    *status = oskar_settings_load(settings, log, filename);

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings->sim.keep_log_file);
    oskar_log_settings_simulator(log, settings);
    oskar_log_settings_observation(log, settings);
    oskar_log_settings_telescope(log, settings);
    oskar_log_settings_beam_pattern(log, settings);

    // Check that an output data file has been specified.
    if (!(settings->beam_pattern.oskar_image_voltage ||
            settings->beam_pattern.oskar_image_phase ||
            settings->beam_pattern.oskar_image_complex ||
            settings->beam_pattern.oskar_image_total_intensity ||
            settings->beam_pattern.fits_image_voltage ||
            settings->beam_pattern.fits_image_phase ||
            settings->beam_pattern.fits_image_total_intensity))
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        oskar_log_error(log, "No output file(s) specified.");
        return;
    }
}

static void simulate_beam_pattern_cube_(oskar_Image* beam_pattern_cube,
        oskar_Settings* settings, int type, oskar_Telescope* tel,
        oskar_Log* log, int* status)
{
    // Make local copies of settings.
    int num_times = settings->obs.num_time_steps;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump = settings->obs.dt_dump_days;
    int station_id = settings->beam_pattern.station_id;
    int* image_size = settings->beam_pattern.size;
    int num_channels = settings->obs.num_channels;
    int num_pols = settings->telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    int num_pixels = 0;
    if (settings->beam_pattern.coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
        num_pixels = image_size[0] * image_size[1];
    else if (settings->beam_pattern.coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_HEALPIX)
    {
        int nside = settings->beam_pattern.nside;
        num_pixels = nside * nside * 12;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return;
    }
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4) beam_pattern_data_type |= OSKAR_MATRIX;

    // Check the station ID is valid.
    if (station_id < 0 || station_id >= oskar_telescope_num_stations(tel)) {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    // Initialise the beam pattern (complex) image cube.
    init_beam_pattern_cube_(beam_pattern_cube, settings, num_pols, status);

    // Initialise temporary array, used to re-order polarisation.
    oskar_Mem* beam_tmp = oskar_mem_create(beam_pattern_data_type,
            OSKAR_LOCATION_CPU, num_pixels, status);

    const oskar_Station* station = oskar_telescope_station_const(tel, station_id);

    // Generate coordinates at which beam the beam pattern is evaluated.
    // This is currently done on the CPU as it is only done once.
    int coord_type = 0;
    oskar_Mem* x = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);
    oskar_Mem* y = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);
    oskar_Mem* z = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);
    oskar_beam_pattern_generate_coordinates(x, y, z, &coord_type,
            oskar_station_beam_longitude_rad(station),
            oskar_station_beam_latitude_rad(station),
            oskar_station_beam_coord_type(station), &settings->beam_pattern,
            status);

    // All GPU memory used within these braces.
    {
        const int GPU = OSKAR_LOCATION_GPU;

        // Set up GPU memory for beam pattern, work array, and coordiantes.
        oskar_StationWork* d_work = oskar_station_work_create(type, GPU, status);
        oskar_Mem* d_beam_pattern = oskar_mem_create(beam_pattern_data_type,
                GPU, num_pixels, status);
        oskar_Station* d_station = oskar_station_create_copy(station, GPU, status);
        oskar_Mem* d_x = oskar_mem_create_copy(x, GPU, status);
        oskar_Mem* d_y = oskar_mem_create_copy(y, GPU, status);
        oskar_Mem* d_z = oskar_mem_create_copy(z, GPU, status);

        // Begin beam pattern evaluation...
        oskar_log_section(log, "Starting simulation...");
        for (int c = 0; c < num_channels; ++c)
        {
            if (*status != OSKAR_SUCCESS) break;

            double frequency = settings->obs.start_frequency_hz +
                    c * settings->obs.frequency_inc_hz;

            oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                    c + 1, num_channels, frequency / 1e6);

            oskar_RandomState* rand_state = oskar_random_state_create(
                    oskar_telescope_max_station_size(tel),
                    oskar_telescope_random_seed(tel), 0, 0, status);

            for (int t = 0; t < num_times; ++t)
            {
                if (*status != OSKAR_SUCCESS) break;

                /* Start time for the data dump, in MJD(UTC). */
                oskar_log_message(log, 1, "Snapshot %4d/%d", t+1, num_times);

                double t_dump = obs_start_mjd_utc + t * dt_dump;
                double GAST = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

                oskar_evaluate_station_beam_pattern(d_beam_pattern, num_pixels,
                        d_x, d_y, d_z, coord_type, d_station, d_work, rand_state,
                        frequency, GAST, status);

                add_beam_pattern_to_complex_cube_(beam_pattern_cube, beam_tmp,
                        d_beam_pattern, num_times, num_pols, num_pixels, t, c,
                        status);
            } // End of time loop

            // Record GPU memory usage.
            oskar_cuda_mem_log(log, 1, 0);

            oskar_random_state_free(rand_state, status);
        } // End of channel loop.
        oskar_station_work_free(d_work, status);
        oskar_station_free(d_station, status);
        oskar_mem_free(d_beam_pattern, status);
        oskar_mem_free(d_x, status);
        oskar_mem_free(d_y, status);
        oskar_mem_free(d_z, status);
        free(d_beam_pattern); // FIXME Remove after updating oskar_mem_free().
        free(d_x); // FIXME Remove after updating oskar_mem_free().
        free(d_y); // FIXME Remove after updating oskar_mem_free().
        free(d_z); // FIXME Remove after updating oskar_mem_free().
    } // GPU memory section

    oskar_mem_free(beam_tmp, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    free(beam_tmp); // FIXME Remove after updating oskar_mem_free().
    free(x); // FIXME Remove after updating oskar_mem_free().
    free(y); // FIXME Remove after updating oskar_mem_free().
    free(z); // FIXME Remove after updating oskar_mem_free().
}

/**
 * @brief
 * Write the single channel & time beam pattern image contained in
 * @p beam_pattern to @p beam_pattern_cube.
 *
 * @details
 * The memory @p beam_cpu is a CPU work array used for re-ordering the beam
 * pattern if needed.
 *
 * Data order in the cube is:
 *
 * [Channel, Time, Polarisation, Declination (y), Right Ascension (x)]
 *
 * Where channel is the slowest varying index.
 *
 * @param beam_pattern_cube Beam pattern image cube.
 * @param beam_temp         Temporary array used for reordering polarisation.
 * @param beam_pattern      Complex beam pattern for a single time / channel.
 * @param num_times         Total number of times in the beam pattern cube.
 * @param num_pols          Total number of polarisations in the beam pattern cube.
 * @param num_pixels        Number of pixels in one image of the beam pattern cube
 * @param t                 The time index of the beam pattern to add.
 * @param c                 The channel index of the beam pattern to add.
 * @param type              The OSKAR data type of the beam pattern.
 *
 * @return An OSKAR error status code.
 */
static void add_beam_pattern_to_complex_cube_(oskar_Image* beam_pattern_cube,
        oskar_Mem* beam_temp, const oskar_Mem* beam_pattern, int num_times,
        int num_pols, int num_pixels, int t, int c, int* status)
{
    if (!status || *status != OSKAR_SUCCESS)
        return;

    int type = oskar_mem_is_double(beam_pattern) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    // Save complex beam pattern data in the right order.
    // Cube has dimension order (from slowest to fastest):
    // Channel, Time, Polarisation, Declination, Right Ascension.
    int offset = (t + c * num_times) * num_pols * num_pixels;
    if (oskar_mem_is_scalar(beam_pattern))
    {
        oskar_mem_insert(&beam_pattern_cube->data, beam_pattern, offset, status);
    }
    else
    {
        // Copy beam pattern back to host memory for re-ordering.
        oskar_mem_copy(beam_temp, beam_pattern, status);
        if (*status != OSKAR_SUCCESS) return;

        // Re-order the polarisation data.
        if (type == OSKAR_SINGLE)
        {
            float2* bp = oskar_mem_float2(&beam_pattern_cube->data, status) + offset;
            float4c* tc = oskar_mem_float4c(beam_temp, status);
            for (int i = 0; i < num_pixels; ++i)
            {
                bp[i]                  = tc[i].a; // theta_X
                bp[i +     num_pixels] = tc[i].b; // phi_X
                bp[i + 2 * num_pixels] = tc[i].c; // theta_Y
                bp[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
        else /* (type == OSKAR_DOUBLE) */
        {
            double2* bp = oskar_mem_double2(&beam_pattern_cube->data, status) + offset;
            double4c* tc = oskar_mem_double4c(beam_temp, status);
            for (int i = 0; i < num_pixels; ++i)
            {
                bp[i]                  = tc[i].a; // theta_X
                bp[i +     num_pixels] = tc[i].b; // phi_X
                bp[i + 2 * num_pixels] = tc[i].c; // theta_Y
                bp[i + 3 * num_pixels] = tc[i].d; // phi_Y
            }
        }
    }
}

static void init_beam_pattern_cube_(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status)
{
    int num_channels = settings->obs.num_channels;
    int num_times = settings->obs.num_time_steps;

    if (settings->beam_pattern.coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        const int* image_size = settings->beam_pattern.size;
        oskar_image_resize(image, image_size[0], image_size[1], num_pols, num_times,
                num_channels, status);
    }
    else if (settings->beam_pattern.coord_grid_type == OSKAR_BEAM_PATTERN_COORDS_HEALPIX)
    {
        int nside = settings->beam_pattern.nside;
        int npix = 12 * nside * nside;
        oskar_image_resize(image, npix, 1, num_pols, num_times, num_channels,
                status);
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS_BEAM_PATTERN;
        return;
    }

    /* Set beam pattern meta-data. */
    image->image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    image->coord_frame        = settings->beam_pattern.coord_frame_type;
    image->grid_type          = settings->beam_pattern.coord_grid_type;
    image->healpix_nside      = settings->beam_pattern.nside;
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

