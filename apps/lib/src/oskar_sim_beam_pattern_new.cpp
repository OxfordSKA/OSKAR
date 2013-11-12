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

#include <apps/lib/oskar_settings_load.h>
#include <apps/lib/oskar_set_up_telescope.h>
#include <apps/lib/oskar_sim_beam_pattern_new.h>

#include <apps/lib/oskar_beam_pattern_generate_coordinates.h>
#include <oskar_evaluate_station_beam_pattern.h>
#include <apps/lib/oskar_beam_pattern_write.h>

#include <oskar_telescope.h>
#include <oskar_image_resize.h>
#include <oskar_image_write.h>
#include <oskar_evaluate_image_lm_grid.h>
#include <oskar_mjd_to_gast_fast.h>
#include <oskar_convert_apparent_ra_dec_to_relative_direction_cosines.h>
#include <oskar_convert_apparent_ra_dec_to_enu_direction_cosines.h>
#include <oskar_evaluate_image_lon_lat_grid.h>
#include <oskar_evaluate_station_beam_aperture_array.h>
#include <oskar_evaluate_station_beam_gaussian.h>
#include <oskar_station_work.h>
#include <oskar_cuda_mem_log.h>
#include <oskar_random_state.h>
#include <oskar_log.h>

#include <oskar_settings_free.h>
#include <oskar_mem_binary_file_write.h>

#include <fits/oskar_fits_image_write.h>

#include <QtCore/QTime>

#include <cmath>
#include <cstdio>      // for remove()


// ============================================================================
static void load_settings_(oskar_Settings* settings, const char* filename,
        oskar_Log* log, int* status);
static void set_up_beam_pattern_(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status);
static void simulate_beam_pattern_cube_(oskar_Image* h_complex_cube,
        oskar_Settings* settings, int type, oskar_Telescope* h_tel,
        oskar_Log* log, int* status);
static void add_beam_pattern_to_complex_cube_(oskar_Image* complex_cube,
        oskar_Mem* beam_cpu, const oskar_Mem* beam_pattern, int num_times,
        int num_pols, int num_pixels, int t, int c, int type, int* status);
// ============================================================================

extern "C"
void oskar_sim_beam_pattern_new(const char* settings_file, oskar_Log* log,
        int* status)
{
    // === Load the settings file. ============================================
    oskar_Settings settings;
    load_settings_(&settings, settings_file, log, status);

    // === Construct/Load the telescope model =================================
    oskar_Telescope* h_tel = oskar_set_up_telescope(log, &settings, status);

    // === Compute the beam pattern cube ======================================
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Image h_complex_cube(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    QTime timer;
    timer.start();
    simulate_beam_pattern_cube_(&h_complex_cube, &settings, type, h_tel, log, status);
    oskar_log_section(log,"Simulation completed in %.3f sec.",timer.elapsed()/1e3);

    // Free memory.
    oskar_telescope_free(h_tel, status);

    // === Write beam pattern to file. ========================================
    oskar_beam_pattern_write(&h_complex_cube, &settings, type, log, status);

    cudaDeviceReset();
}

static void load_settings_(oskar_Settings* settings, const char* filename,
        oskar_Log* log, int* status)
{
    oskar_log_section(log, "Loading settings file '%s'", filename);
    *status = oskar_settings_load(settings, log, filename);
    if (*status != OSKAR_SUCCESS) return;

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
        oskar_log_error(log, "No output file(s) specified.");
        return;
    }
}

static void simulate_beam_pattern_cube_(oskar_Image* h_complex_cube,
        oskar_Settings* settings, int type, oskar_Telescope* h_tel,
        oskar_Log* log, int* status)
{
    // Local copies of settings.
    int num_times = settings->obs.num_time_steps;
    double obs_start_mjd_utc = settings->obs.start_mjd_utc;
    double dt_dump = settings->obs.dt_dump_days;
    int station_id = settings->beam_pattern.station_id;
    int* image_size = settings->beam_pattern.size;
    int num_channels = settings->obs.num_channels;
    int num_pols = settings->telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    int num_pixels = image_size[0] * image_size[1];
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4) beam_pattern_data_type |= OSKAR_MATRIX;

    // Check station ID is set_up_beam_patternwithin range.
    if (station_id < 0 || station_id >= oskar_telescope_num_stations(h_tel)) {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return;
    }

    // Set up beam pattern (complex) image cube.
    set_up_beam_pattern_(h_complex_cube, settings, num_pols, status);

    // Temporary CPU (host) memory, used to re-order polarisation data.
    oskar_Mem h_beam_temp;
    oskar_mem_init(&h_beam_temp, beam_pattern_data_type, OSKAR_LOCATION_CPU,
            num_pixels, OSKAR_TRUE, status);

    /* CPU (host) arrays for beam pattern coordinates */
    int coord_type = 0;
    oskar_Mem* x = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);
    oskar_Mem* y = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);
    oskar_Mem* z = oskar_mem_create(type, OSKAR_LOCATION_CPU, num_pixels, status);

    /* Generate beam pattern coordinates on the CPU */
    /* Note: This only has to be done once as the coordinate grid is constant */
    oskar_beam_pattern_generate_coordinates(x, y, z, &coord_type,
            &settings->beam_pattern, status);

    // All GPU memory used within these braces.
    {
        const int GPU = OSKAR_LOCATION_GPU;

        // Initialise work array and GPU memory for a beam pattern.
        oskar_StationWork* d_work = oskar_station_work_create(type, GPU, status);
        oskar_Mem d_beam_pattern;
        oskar_mem_init(&d_beam_pattern, beam_pattern_data_type, GPU, num_pixels,
                OSKAR_TRUE, status);

        // Copy station data to GPU memory.
        const oskar_Station* station = oskar_telescope_station_const(h_tel, station_id);
        oskar_Station* d_station = oskar_station_create_copy(station, GPU, status);

        oskar_Mem* d_x = oskar_mem_create_copy(x, GPU, status);
        oskar_Mem* d_y = oskar_mem_create_copy(y, GPU, status);
        oskar_Mem* d_z = oskar_mem_create_copy(z, GPU, status);

        // === Loop over channels and time to evaluate patterns. ===============
        oskar_log_section(log, "Starting simulation...");
        for (int c = 0; c < num_channels; ++c)
        {
            double frequency = settings->obs.start_frequency_hz +
                    c * settings->obs.frequency_inc_hz;
            oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                    c + 1, num_channels, frequency / 1e6);

            // Create random number state.
            // Note: this will be seeded the same per channel ... is this correct?
            oskar_RandomState* random_state = oskar_random_state_create(
                    oskar_telescope_max_station_size(h_tel),
                    oskar_telescope_random_seed(h_tel), 0, 0, status);

            for (int t = 0; t < num_times; ++t)
            {
                /* Check error code. */
                if (*status != OSKAR_SUCCESS) continue;

                /* Start time for the data dump, in MJD(UTC). */
                oskar_log_message(log, 1, "Snapshot %4d/%d", t+1, num_times);

                double t_dump = obs_start_mjd_utc + t * dt_dump;
                double GAST = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);


                oskar_evaluate_station_beam_pattern(&d_beam_pattern, num_pixels,
                        d_x, d_y, d_z, coord_type, d_station, d_work,
                        random_state, frequency, GAST, status);

                /* Copy complex beam pattern data to host array with reordering. */
                add_beam_pattern_to_complex_cube_(h_complex_cube, &h_beam_temp,
                        &d_beam_pattern, num_times, num_pols, num_pixels, t, c,
                        type, status);

            } // End of time loop

            // Record GPU memory usage.
            oskar_cuda_mem_log(log, 1, 0);

            // Free memory.
            oskar_random_state_free(random_state, status);
        } // End of channel loop.

        // === Free memory. ===================================================
        oskar_station_work_free(d_work, status);
        oskar_mem_free(&d_beam_pattern, status);
        oskar_station_free(d_station, status);
        oskar_mem_free(d_x, status);
        oskar_mem_free(d_y, status);
        oskar_mem_free(d_z, status);
    } // GPU memory section

    oskar_mem_free(&h_beam_temp, status);
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
}

/* TODO !!!DEPRECATED!!
// Note: this function assumes d_coords work array has been correctly populated!
static void generate_beam_pattern_(oskar_Mem* d_beam_pattern,
        oskar_Station* d_station, oskar_StationBeamCoords* d_coords,
        oskar_StationWork* d_work, oskar_RandomState* rand_state,
        size_t num_pixels, double frequency, double gast, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    switch (oskar_station_station_type(d_station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            // Evaluates the AA station beam for the given station structure,
            // and frequency and time for the provided horizontal coordinates.
            // Note: the beam phase centre is
            oskar_evaluate_station_beam_aperture_array(d_beam_pattern,
                    d_station, num_pixels, &d_coords->enu_direction_x,
                    &d_coords->enu_direction_y, &d_coords->enu_direction_z, gast,
                    frequency, d_work, rand_state, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            double fwhm_rad = oskar_station_gaussian_beam_fwhm_rad(d_station);
            oskar_evaluate_station_beam_gaussian(d_beam_pattern, num_pixels,
                    &d_coords->relative_direction_l, &d_coords->relative_direction_m,
                    &d_coords->enu_direction_z, fwhm_rad, status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    };
}
*/

/**
 * @brief
 * Write the beam pattern contained in @p beam_pattern to @p complex_cube.
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
 * @param complex_cube
 * @param beam_cpu
 * @param beam_pattern
 * @param num_times
 * @param num_pols
 * @param num_pixels
 * @param t
 * @param c
 * @param type
 *
 * @return An OSKAR error status code.
 */
static void add_beam_pattern_to_complex_cube_(oskar_Image* complex_cube,
        oskar_Mem* beam_cpu, const oskar_Mem* beam_pattern, int num_times,
        int num_pols, int num_pixels, int t, int c, int type, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    // Save complex beam pattern data in the right order.
    // Cube has dimension order (from slowest to fastest):
    // Channel, Time, Polarisation, Declination, Right Ascension.
    int offset = (t + c * num_times) * num_pols * num_pixels;
    if (oskar_mem_is_scalar(beam_pattern))
    {
        oskar_mem_insert(&complex_cube->data, beam_pattern, offset, status);
    }
    else
    {
        // Copy beam pattern back to host memory for re-ordering.
        oskar_mem_copy(beam_cpu, beam_pattern, status);
        if (*status != OSKAR_SUCCESS) return;

        // Re-order the polarisation data.
        if (type == OSKAR_SINGLE)
        {
            float2* bp = (float2*)complex_cube->data.data + offset;
            float4c* tc = (float4c*)beam_cpu->data;
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
            double2* bp = (double2*)complex_cube->data.data + offset;
            double4c* tc = (double4c*)beam_cpu->data;
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

static void set_up_beam_pattern_(oskar_Image* image,
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

