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
#include "apps/lib/oskar_sim_beam_pattern_new.h"

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

#include <cmath>
#include <cstdio>      // for remove()
#include <iostream>    // debugging

#include <QtCore/QTime>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
struct coords3D {
    oskar_Mem x;
    oskar_Mem y;
    oskar_Mem z;
};
struct station_beam_coords {
    coords3D horizontal;
    coords3D relative;
    oskar_Mem RA;
    oskar_Mem Dec;
    oskar_Mem radius;
};
static void init_coords_3D(coords3D& coords, int type, int location,
        size_t num_elements, int owner, int* status);
static void free_coords_3D(coords3D& coords, int* status);
static void init_station_beam_coords(station_beam_coords& coords,
        int type, int location, size_t num_elements, int owner, int* status);
static void free_station_beam_coords(station_beam_coords& coords, int* status);
static int load_settings(oskar_Settings& settings, const char* filename,
        oskar_Log* log);
static void set_up_beam_pattern(oskar_Image* image,
        const oskar_Settings* settings, int num_pols, int* status);
static int evaluate_station_beam(oskar_Image& h_complex_cube,
        oskar_Settings& settings, int type, oskar_Telescope* h_tel,
        oskar_Log* log);
static void get_beam_coords_(station_beam_coords& d_coords,
        oskar_Station& d_station, oskar_Settings& settings,
        double gast, int* status);
static void coords_phase_centre_tangent_plane_direction_(
        station_beam_coords& d_coords, oskar_Station& d_station,
        oskar_Settings& settings, double gast, int* status);
static void coords_horizon_direction(station_beam_coords& d_coords,
        oskar_Station& d_station, oskar_Settings& settings,
        double gast, int* status);
static void make_beam_(oskar_Mem& d_beam_pattern,
        oskar_Station& d_station, station_beam_coords& d_coords,
        oskar_StationWork& d_work, oskar_RandomState& rand_state,
        size_t num_pixels, double frequency, double gast, int* status);
static void add_to_cube_(oskar_Image& complex_cube, oskar_Mem& beam_cpu,
        const oskar_Mem& beam_pattern, int num_times, int num_pols,
        int num_pixels, int t, int c, int type, int* status);
static int save_output(const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log);
static void save_complex(const oskar_Image& complex_cube,
        const oskar_Settings& settings, oskar_Log* log, int* status);
static void save_power(oskar_Image& image_cube,
        const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log,
        int num_pixels_total, int* err);
static void save_phase(const oskar_Image& complex_cube,
        oskar_Image& image_cube, const oskar_Settings& settings, int type,
        oskar_Log* log, int num_pixels_total, int* err);
static void save_total_intensity(const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log, int* status);
// ============================================================================

extern "C"
int oskar_sim_beam_pattern_new(const char* settings_file, oskar_Log* log)
{
    int err = OSKAR_SUCCESS;

    // === Load the settings file. ============================================
    oskar_Settings settings;
    err = load_settings(settings, settings_file, log);
    if (err) return err;

    // === Construct/Load the telescope model =================================
    oskar_Telescope* h_tel = oskar_set_up_telescope(log, &settings, &err);
    if (err) return err;

    // === Compute the beam pattern cube ======================================
    int type = settings.sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_Image h_complex_cube(type | OSKAR_COMPLEX, OSKAR_LOCATION_CPU);
    QTime timer;
    timer.start();
    err = evaluate_station_beam(h_complex_cube, settings, type, h_tel, log);
    oskar_log_section(log, "Simulation completed in %.3f sec.",
            timer.elapsed() / 1e3);
    if (err) return err;

    // Free memory.
    oskar_telescope_free(h_tel, &err);

    // === Write beam pattern to file. ========================================
    err = save_output(h_complex_cube, settings, type, log);
    if (err) return err;

    cudaDeviceReset();
    return err;
}

static int load_settings(oskar_Settings& settings, const char* filename,
        oskar_Log* log)
{
    int err = OSKAR_SUCCESS;

    oskar_log_section(log, "Loading settings file '%s'", filename);
    err = oskar_settings_load(&settings, log, filename);
    if (err) return err;

    // Log the relevant settings.
    oskar_log_set_keep_file(log, settings.sim.keep_log_file);
    oskar_log_settings_simulator(log, &settings);
    oskar_log_settings_observation(log, &settings);
    oskar_log_settings_telescope(log, &settings);
    oskar_log_settings_beam_pattern(log, &settings);

    // Check that an output data file has been specified.
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

    return err;
}

static int evaluate_station_beam(oskar_Image& h_complex_cube,
        oskar_Settings& settings, int type, oskar_Telescope* h_tel,
        oskar_Log* log)
{
    int err = OSKAR_SUCCESS;

    // Local copies of settings.
    int num_times = settings.obs.num_time_steps;
    double obs_start_mjd_utc = settings.obs.start_mjd_utc;
    double dt_dump = settings.obs.dt_dump_days;
    int station_id = settings.beam_pattern.station_id;
    int* image_size = settings.beam_pattern.size;
    int num_channels = settings.obs.num_channels;
    int num_pols = settings.telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    int num_pixels = image_size[0] * image_size[1];
    int beam_pattern_data_type = type | OSKAR_COMPLEX;
    if (num_pols == 4) beam_pattern_data_type |= OSKAR_MATRIX;

    // Check station ID is within range.
    if (station_id < 0 || station_id >= oskar_telescope_num_stations(h_tel))
        return OSKAR_ERR_OUT_OF_RANGE;

    // Set up beam pattern (complex) image cube.
    set_up_beam_pattern(&h_complex_cube, &settings, num_pols, &err);
    if (err) return err;

    // Temporary CPU (host) memory, used to re-order polarisation data.
    // XXX move this into to work array?
    oskar_Mem h_beam_temp;
    oskar_mem_init(&h_beam_temp, beam_pattern_data_type, OSKAR_LOCATION_CPU,
            num_pixels, OSKAR_TRUE, &err);

    // All GPU memory used within these braces.
    {
        const int gpu = OSKAR_LOCATION_GPU;

        // Initialise work array and GPU memory for a beam pattern.
        oskar_StationWork* d_work = oskar_station_work_create(type, gpu, &err);
        oskar_Mem d_beam_pattern;
        oskar_mem_init(&d_beam_pattern, beam_pattern_data_type, gpu, num_pixels,
                OSKAR_TRUE, &err);

        // Copy station data to GPU memory.
        const oskar_Station* station = oskar_telescope_station_const(h_tel, station_id);
        oskar_Station* d_station = oskar_station_create_copy(station, gpu, &err);

        // Initialise local station beam coordinates work structure.
        station_beam_coords d_coords;
        init_station_beam_coords(d_coords, type, gpu, num_pixels, 1, &err);

        // === Loop over channels and time to evaluate patterns. ===============
        oskar_log_section(log, "Starting simulation...");
        for (int c = 0; c < num_channels; ++c)
        {
            double frequency = settings.obs.start_frequency_hz +
                    c * settings.obs.frequency_inc_hz;
            oskar_log_message(log, 0, "Channel %3d/%d [%.4f MHz]",
                    c + 1, num_channels, frequency / 1e6);

            // Create random number state.
            // Note: this will be seeded the same per channel ... is this correct?
            oskar_RandomState* random_state = oskar_random_state_create(
                    oskar_telescope_max_station_size(h_tel),
                    oskar_telescope_random_seed(h_tel), 0, 0, &err);

            for (int t = 0; t < num_times; ++t)
            {
                // Check error code.
                if (err) continue;

                // Start time for the data dump, in MJD(UTC).
                oskar_log_message(log, 1, "Snapshot %4d/%d", t+1, num_times);

                double t_dump = obs_start_mjd_utc + t * dt_dump;
                double gast = oskar_mjd_to_gast_fast(t_dump + dt_dump / 2.0);

                // Evaluate the station beam coordinates.
                get_beam_coords_(d_coords, *d_station, settings, gast, &err);

                // Evaluate the station beam.
                make_beam_(d_beam_pattern, *d_station, d_coords,
                        *d_work, *random_state, num_pixels, frequency,
                        gast, &err);

                // Copy complex beam pattern data to host array with reordering.
                add_to_cube_(h_complex_cube, h_beam_temp, d_beam_pattern,
                        num_times, num_pols, num_pixels, t, c, type, &err);

            } // End of time loop

            // Record GPU memory usage.
            oskar_cuda_mem_log(log, 1, 0);

            // Free memory.
            oskar_random_state_free(random_state, &err);
        } // End of channel loop.

        // === Free memory. ===================================================
        oskar_station_work_free(d_work, &err);
        oskar_mem_free(&d_beam_pattern, &err);
        oskar_mem_free(&h_beam_temp, &err);
        oskar_station_free(d_station, &err);
        free_station_beam_coords(d_coords, &err);
    } // GPU memory section

    return err;
}

/**
 * @brief
 * This function converts from beam coordinates specified in the settings to
 * the coordinates required by the beam evaluation function of the specified
 * station type.
 *
 * Notes:
 * Separate grid generation from coordinate system from conversion.
 * Need to convert from:
 * - RA, Dec list
 * - Alt, Az list
 * - HEALPix [RA, Dec] (same as RA, Dec list)
 * - HEALPix [Alt, Az] (same as Alt, Az list)
 * - List of direction cosines w.r.t. phase centre
 * - List of direction cosines w.r.t. local zenith
 *
 * Need to convert to:
 * - Radius form phase centre
 * - Direction cosines relative to phase centre.
 * - Direction cosines relative to the local zenith.
 * - Alt, Az
 * - RA, Dec
 * - Horizontal x,y,z
 *
 * Questions:
 * - is horizontal x,y,z different from horizontal l,m,n ?!
 *
 * @param d_coords
 * @param d_station
 * @param settings
 * @param gast
 * @param status
 */
static void get_beam_coords_(station_beam_coords& d_coords,
        oskar_Station& d_station, oskar_Settings& settings, double gast,
        int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    int settings_coord_grid_type = 0;
    switch (settings_coord_grid_type)
    {
        case 0: // Grid specified as FOV and number of pixels relative to phase
                // centre.
        {
            coords_phase_centre_tangent_plane_direction_(d_coords, d_station,
                    settings, gast, status);
            break;
        }
        case 1: // Grid specified by a FOV and number of pixels relative to the
                // local station zenith.
        {
            coords_horizon_direction(d_coords, d_station, settings, gast, status);
            break;
        }
        case 2: // HEALPix grid.
        {
            break;
        }
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    };
}

/**
 * @brief
 * Evaluate beam coordinates for an output beam image specified
 * as a tangent plane grid relative to the beam pointing direction.
 *
 * @details
 * Generate coordinates required for each beam evaluation function when the
 * beam image output coordinates are specified as a grid of pixels on a tangent
 * plane relative to the beam phase centre.
 *
 * Note: Structures passed to this function with the prefix 'd_' are
 *       assumed to be located in Device (GPU) memory.
 *
 * @param[out]    d_coords      Coordinates work array structure.
 * @param[in]     d_station     Station structure.
 * @param[in]     settings      Settings structure.
 * @param[in]     gast          Greenwich apparent siderial time, in radians.
 * @param[in/out] status        Error status code.
 */
static void coords_phase_centre_tangent_plane_direction_(
        station_beam_coords& d_coords, oskar_Station& d_station,
        oskar_Settings& settings, double gast, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    // Local copies of settings.
    double lat     = oskar_station_latitude_rad(&d_station);
    double lon     = oskar_station_longitude_rad(&d_station);
    double ra0     = oskar_station_beam_longitude_rad(&d_station);
    double dec0    = oskar_station_beam_latitude_rad(&d_station);
    double fov_lon = settings.beam_pattern.fov_deg[0];
    double fov_lat = settings.beam_pattern.fov_deg[1];
    int* size      = settings.beam_pattern.size;
    double last    = gast + lon;
    int num_pixels = size[0] * size[1];

    // Generate coordinate grid (TODO only needs to be done once!)
    // TODO Correct name for this function (lon_lat, ra_dec, theta_phi?)
    oskar_evaluate_image_lon_lat_grid(&d_coords.RA, &d_coords.Dec,
            size[0], size[1], fov_lon, fov_lat, ra0, dec0, status);

    switch (oskar_station_station_type(&d_station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            // Convert to horizontal x,y,z as the AA beam evaluation
            // requires this coordinate system.
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines(num_pixels,
                    &d_coords.horizontal.x, &d_coords.horizontal.y,
                    &d_coords.horizontal.z, &d_coords.RA,
                    &d_coords.Dec, last, lat, status);
            break;
        }

        // Convert to relative x,y,z and horizontal z (for horizon clip)
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            // Need horizon directions for horizon (z) clip.
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines(num_pixels,
                    &d_coords.horizontal.x, &d_coords.horizontal.y,
                    &d_coords.horizontal.z, &d_coords.RA,
                    &d_coords.Dec, last, lat, status);
            oskar_convert_apparent_ra_dec_to_relative_direction_cosines(num_pixels,
                    &d_coords.RA, &d_coords.Dec, ra0, dec0,
                    &d_coords.relative.x, &d_coords.relative.y,
                    &d_coords.relative.z, status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
            // populate radius
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    };
}

static void coords_horizon_direction(station_beam_coords& d_coords,
        oskar_Station& d_station, oskar_Settings& settings,
        double /*gast*/, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    /*double lat = oskar_station_latitude_rad(&d_station);*/
    /*double lon = oskar_station_longitude_rad(&d_station);*/
    /*double last = gast + lon;*/
    double ra0 = 0;
    double dec0 = M_PI/2.0;
    double fov_lon = settings.beam_pattern.fov_deg[0];
    double fov_lat = settings.beam_pattern.fov_deg[1];
    int* image_size = settings.beam_pattern.size;
    int num_pixels = image_size[0] * image_size[1];
    /*int type = oskar_station_type(&d_station);*/
    /*const int cpu = OSKAR_LOCATION_CPU;*/

    // Generate coordinate grid (XXX only needs to be done once!)
    oskar_evaluate_image_lon_lat_grid(&d_coords.RA, &d_coords.Dec,
            image_size[0], image_size[1], fov_lon, fov_lat, ra0, dec0,
            status);

    switch (oskar_station_station_type(&d_station))
    {
        // Convert to horizontal x,y,z
        case OSKAR_STATION_TYPE_AA:
        {
            oskar_convert_apparent_ra_dec_to_enu_direction_cosines(num_pixels,
                    &d_coords.horizontal.x, &d_coords.horizontal.y,
                    &d_coords.horizontal.z, &d_coords.RA,
                    &d_coords.Dec, 0, dec0, status);
            break;
        }

        // Convert to relative l,m,m and horizontal z (for horizon clip)
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
            // populate radius
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    };
}

// Note: this function assumes d_coords work array has been correctly populated!
static void make_beam_(oskar_Mem& d_beam_pattern,
        oskar_Station& d_station, station_beam_coords& d_coords,
        oskar_StationWork& d_work, oskar_RandomState& rand_state,
        size_t num_pixels, double frequency, double gast, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    switch (oskar_station_station_type(&d_station))
    {
        case OSKAR_STATION_TYPE_AA:
        {
            // Evaluates the AA station beam for the given station structure,
            // and frequency and time for the provided horizontal coordinates.
            // Note: the beam phase centre is
            oskar_evaluate_station_beam_aperture_array(&d_beam_pattern,
                    &d_station, num_pixels, &d_coords.horizontal.x,
                    &d_coords.horizontal.y, &d_coords.horizontal.z, gast,
                    frequency, &d_work, &rand_state, status);
            break;
        }
        case OSKAR_STATION_TYPE_GAUSSIAN_BEAM:
        {
            double fwhm_rad = oskar_station_gaussian_beam_fwhm_rad(&d_station);
            oskar_evaluate_station_beam_gaussian(&d_beam_pattern, num_pixels,
                    &d_coords.relative.x, &d_coords.relative.y,
                    &d_coords.horizontal.z, fwhm_rad, status);
            break;
        }
        case OSKAR_STATION_TYPE_VLA_PBCOR:
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            break;
    };
}

/**
 * @brief Write the beam pattern contained in @p beam_pattern to @p complex_cube.
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
static void add_to_cube_(oskar_Image& complex_cube, oskar_Mem& beam_cpu,
        const oskar_Mem& beam_pattern, int num_times, int num_pols,
        int num_pixels, int t, int c, int type, int* status)
{
    if (!status || *status != OSKAR_SUCCESS) return;

    // Save complex beam pattern data in the right order.
    // Cube has dimension order (from slowest to fastest):
    // Channel, Time, Polarisation, Declination, Right Ascension.
    int offset = (t + c * num_times) * num_pols * num_pixels;
    if (oskar_mem_is_scalar(&beam_pattern))
    {
        oskar_mem_insert(&complex_cube.data, &beam_pattern, offset, status);
    }
    else
    {
        // Copy beam pattern back to host memory for re-ordering.
        oskar_mem_copy(&beam_cpu, &beam_pattern, status);
        if (*status != OSKAR_SUCCESS) return;

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
}

static void set_up_beam_pattern(oskar_Image* image,
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

/**
 * @brief Save beam patterns to file.
 *
 * @param complex_cube
 * @param image_cube
 * @param settings
 * @param type
 * @param log
 * @param num_pixels_total
 * @param err
 */
static int save_output(const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log)
{
    int err = OSKAR_SUCCESS;

    // Set up image cube for beam pattern output images.
    const int* image_size = settings.beam_pattern.size;
    int num_times = settings.obs.num_time_steps;
    int num_channels = settings.obs.num_channels;
    int num_pols = settings.telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    int num_pixels = image_size[0] * image_size[1];
    int num_pixels_total = num_pixels * num_times * num_channels * num_pols;
    oskar_Image image_cube(type, OSKAR_LOCATION_CPU);
    set_up_beam_pattern(&image_cube, &settings, num_pols, &err);
    if (err) return err;

    // Save the complex beam pattern.
    save_complex(complex_cube, settings, log, &err);
    if (err) return err;

    // Save the power beam pattern.
    save_power(image_cube, complex_cube, settings, type, log, num_pixels_total,
            &err);
    if (err) return err;

    // Save the phase beam pattern.
    save_phase(complex_cube, image_cube, settings, type, log, num_pixels_total,
            &err);
    if (err) return err;

    // Save the total intensity beam pattern.
    save_total_intensity(complex_cube, settings, type, log, &err);
    if (err) return err;

    return err;
}

/**
 * @brief Save the complex beam pattern.
 *
 * @param complex_cube
 * @param settings
 * @param log
 * @param err
 */
static void save_complex(const oskar_Image& complex_cube,
        const oskar_Settings& settings, oskar_Log* log, int* err)
{
    const char* filename = settings.beam_pattern.oskar_image_complex;
    // Return if there is an error or the complex cube filename has not been set.
    if ((err && *err != OSKAR_SUCCESS) || !filename)
        return;
    oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
    oskar_image_write(&complex_cube, log, filename, 0, err);
}

/**
 * @brief Save the power beam pattern.
 *
 * @param image_cube
 * @param complex_cube
 * @param settings
 * @param type
 * @param log
 * @param num_pixels_total
 * @param err
 */
static void save_power(oskar_Image& image_cube,
        const oskar_Image& complex_cube,
        const oskar_Settings& settings, int type, oskar_Log* log,
        int num_pixels_total, int* err)
{
    const char* filename;

    // Write out power data if required.
    if (settings.beam_pattern.oskar_image_voltage ||
            settings.beam_pattern.fits_image_voltage)
    {
        // Convert complex values to power (amplitude of complex number).
        if (type == OSKAR_SINGLE)
        {
            float* image_data = oskar_mem_float(&image_cube.data, err);
            const float2* complex_data = oskar_mem_float2_const(&complex_cube.data, err);
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
            double* image_data = oskar_mem_double(&image_cube.data, err);
            const double2* complex_data = oskar_mem_double2_const(&complex_cube.data, err);
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
        if (filename && !*err)
        {
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(&image_cube, log, filename, 0, err);
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        filename = settings.beam_pattern.fits_image_voltage;
        if (filename && !*err)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(&image_cube, log, filename, err);
        }
#endif
    }
}

static void save_phase(const oskar_Image& complex_cube,
        oskar_Image& image_cube, const oskar_Settings& settings, int type,
        oskar_Log* log, int num_pixels_total, int* err)
{
    const char* filename;

    // Write out phase data if required.
    if (settings.beam_pattern.oskar_image_phase ||
            settings.beam_pattern.fits_image_phase)
    {
        // Convert complex values to phase.
        if (type == OSKAR_SINGLE)
        {
            float* image_data = oskar_mem_float(&image_cube.data, err);
            const float2* complex_data = oskar_mem_float2_const(&complex_cube.data, err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data = oskar_mem_double(&image_cube.data, err);
            const double2* complex_data = oskar_mem_double2_const(&complex_cube.data, err);
            for (int i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }

        // Write OSKAR image.
        filename = settings.beam_pattern.oskar_image_phase;
        if (filename && !*err)
        {
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(&image_cube, log, filename, 0, err);
        }
#ifndef OSKAR_NO_FITS
        // Write FITS image.
        filename = settings.beam_pattern.fits_image_phase;
        if (filename && !*err)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(&image_cube, log, filename, err);
        }
#endif
    }
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
    set_up_beam_pattern(&image_cube_I, &settings, 1, status);

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

static void init_coords_3D(coords3D& coords, int type, int location,
        size_t num_elements, int owner, int* status)
{
    oskar_mem_init(&coords.x, type, location, num_elements, owner, status);
    oskar_mem_init(&coords.y, type, location, num_elements, owner, status);
    oskar_mem_init(&coords.z, type, location, num_elements, owner, status);
}

static void free_coords_3D(coords3D& coords, int* status)
{
    oskar_mem_free(&coords.x, status);
    oskar_mem_free(&coords.y, status);
    oskar_mem_free(&coords.z, status);
}

static void init_station_beam_coords(station_beam_coords& coords,
        int type, int location, size_t num_elements, int owner, int* status)
{
    init_coords_3D(coords.horizontal, type, location, num_elements, owner, status);
    init_coords_3D(coords.relative, type, location, num_elements, owner, status);
    oskar_mem_init(&coords.RA, type, location, num_elements, owner, status);
    oskar_mem_init(&coords.Dec, type, location, num_elements, owner, status);
    oskar_mem_init(&coords.radius, type, location, num_elements, owner, status);
}

static void free_station_beam_coords(station_beam_coords& coords, int* status)
{
    free_coords_3D(coords.horizontal, status);
    free_coords_3D(coords.relative, status);
    oskar_mem_free(&coords.RA, status);
    oskar_mem_free(&coords.Dec, status);
    oskar_mem_free(&coords.radius, status);

}
