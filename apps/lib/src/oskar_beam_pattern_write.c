/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <apps/lib/oskar_beam_pattern_write.h>
#include <oskar_telescope.h>
#include <oskar_image.h>
#include <fits/oskar_fits_image_write.h>
#include <oskar_cmath.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
static void save_complex(const oskar_Image* complex_cube,
        const oskar_Settings* settings, oskar_Log* log, int* status);
static void save_voltage(oskar_Image* image_cube,
        const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log,
        int num_pixels_total, int* err);
static void save_phase(const oskar_Image* complex_cube,
        oskar_Image* image_cube, const oskar_Settings* settings, int type,
        oskar_Log* log, int num_pixels_total, int* err);
static void save_total_intensity(const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log, int* status);
/* ========================================================================== */

void oskar_beam_pattern_write(const oskar_Image* complex_cube,
        oskar_Settings* settings, int type, oskar_Log* log, int* status)
{
    /* Set up image cube for beam pattern output images. */
    int num_times, num_channels, num_pols, num_pixels = 0, num_pixels_total;
    oskar_Image* image;

    if (!status || *status != OSKAR_SUCCESS)
        return;

    if (!complex_cube || !settings || !log) {
        *status = OSKAR_ERR_INVALID_ARGUMENT;
        return;
    }

    image = oskar_image_create(type, OSKAR_CPU, status);

    num_times = settings->beam_pattern.time_average_beam ? 1 :
            settings->obs.num_time_steps;
    num_channels = settings->obs.num_channels;
    num_pols = settings->telescope.pol_mode == OSKAR_POL_MODE_FULL ? 4 : 1;
    if (settings->beam_pattern.coord_grid_type ==
            OSKAR_BEAM_PATTERN_COORDS_BEAM_IMAGE)
    {
        const int* size = settings->beam_pattern.size;
        num_pixels = size[0] * size[1];
        oskar_image_resize(image, size[0], size[1], num_pols, num_times,
                num_channels, status);
    }
    else if (settings->beam_pattern.coord_grid_type ==
            OSKAR_BEAM_PATTERN_COORDS_HEALPIX)
    {
        int nside = settings->beam_pattern.nside;
        num_pixels = 12*nside*nside;
        oskar_image_resize(image, num_pixels, 1, num_pols, num_times,
                num_channels, status);
    }

    /* Set meta-data */
    oskar_image_set_type(image, (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED);
    oskar_image_set_coord_frame(image, settings->beam_pattern.coord_frame_type);
    oskar_image_set_grid_type(image, settings->beam_pattern.coord_grid_type);
    oskar_image_set_healpix_nside(image, settings->beam_pattern.nside);
    oskar_image_set_centre(image,
            settings->obs.phase_centre_lon_rad[0] * 180.0 / M_PI,
            settings->obs.phase_centre_lat_rad[0] * 180.0 / M_PI);
    oskar_image_set_fov(image, settings->beam_pattern.fov_deg[0],
            settings->beam_pattern.fov_deg[1]);
    oskar_image_set_freq(image, settings->obs.start_frequency_hz,
            settings->obs.frequency_inc_hz);
    oskar_image_set_time(image, settings->obs.start_mjd_utc,
            settings->obs.dt_dump_days * 86400.0);
    oskar_mem_append_raw(oskar_image_settings_path(image),
            settings->settings_path, OSKAR_CHAR, OSKAR_CPU,
            1 + strlen(settings->settings_path), status);

    num_pixels_total = num_pixels * num_times * num_channels * num_pols;

    /* Save the complex beam pattern. */
    save_complex(complex_cube, settings, log, status);

    /* Save the power beam pattern. */
    save_voltage(image, complex_cube, settings, type, log, num_pixels_total,
            status);

    /* Save the phase beam pattern. */
    save_phase(complex_cube, image, settings, type, log, num_pixels_total,
            status);

    /* Save the total intensity beam pattern. */
    save_total_intensity(complex_cube, settings, type, log, status);

    oskar_image_free(image, status);
}

static void save_complex(const oskar_Image* complex_cube,
        const oskar_Settings* settings, oskar_Log* log, int* status)
{
    const char* filename = settings->beam_pattern.oskar_image_complex;

    /* Return if there is an error or the filename has not been set. */
    if ((status && *status != OSKAR_SUCCESS) || !filename)
        return;
    oskar_log_message(log, 'M', 0, "Writing OSKAR image file: '%s'", filename);
    oskar_image_write(complex_cube, log, filename, 0, status);
}

static void save_voltage(oskar_Image* image_cube,
        const oskar_Image* complex_cube, const oskar_Settings* settings,
        int type, oskar_Log* log,
        int num_pixels_total, int* status)
{
    const char* filename;
    int i;

    /* Write out power data if required. */
    if (settings->beam_pattern.oskar_image_voltage ||
            settings->beam_pattern.fits_image_voltage)
    {
        /* Convert complex values to power (amplitude of complex number). */
        if (type == OSKAR_SINGLE)
        {
            float* image_data;
            const float2* complex_data;
            image_data = oskar_mem_float(oskar_image_data(image_cube), status);
            complex_data = oskar_mem_float2_const(
                    oskar_image_data_const(complex_cube), status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                float x, y;
                x = complex_data[i].x;
                y = complex_data[i].y;
                image_data[i] = sqrt(x*x + y*y);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data;
            const double2* complex_data;
            image_data = oskar_mem_double(oskar_image_data(image_cube), status);
            complex_data = oskar_mem_double2_const(
                    oskar_image_data_const(complex_cube), status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                double x, y;
                x = complex_data[i].x;
                y = complex_data[i].y;
                image_data[i] = sqrt(x*x + y*y);
            }
        }

        /* Write OSKAR image. */
        filename = settings->beam_pattern.oskar_image_voltage;
        if (filename && !*status)
        {
            oskar_log_message(log, 'M', 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(image_cube, log, filename, 0, status);
        }

        /* Write FITS image. */
        filename = settings->beam_pattern.fits_image_voltage;
        if (filename && !*status)
        {
            oskar_log_message(log, 'M', 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(image_cube, log, filename, status);
        }
    }
}

static void save_phase(const oskar_Image* complex_cube,
        oskar_Image* image_cube, const oskar_Settings* settings, int type,
        oskar_Log* log, int num_pixels_total, int* status)
{
    const char* filename;
    int i;

    /* Write out phase data if required. */
    if (settings->beam_pattern.oskar_image_phase ||
            settings->beam_pattern.fits_image_phase)
    {
        /* Convert complex values to phase. */
        if (type == OSKAR_SINGLE)
        {
            float* image_data;
            const float2* complex_data;
            image_data = oskar_mem_float(oskar_image_data(image_cube), status);
            complex_data = oskar_mem_float2_const(
                    oskar_image_data_const(complex_cube), status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data;
            const double2* complex_data;
            image_data = oskar_mem_double(oskar_image_data(image_cube), status);
            complex_data = oskar_mem_double2_const(
                    oskar_image_data_const(complex_cube), status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }

        /* Write OSKAR image. */
        filename = settings->beam_pattern.oskar_image_phase;
        if (filename && !*status)
        {
            oskar_log_message(log, 'M', 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(image_cube, log, filename, 0, status);
        }

        /* Write FITS image. */
        filename = settings->beam_pattern.fits_image_phase;
        if (filename && !*status)
        {
            oskar_log_message(log, 'M', 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(image_cube, log, filename, status);
        }
    }
}

static void save_total_intensity(const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log, int* status)
{
    const char* filename;
    int num_channels, num_times, num_pols, num_pixels;
    int c, t, p, i, idx, islice, width, height;
    double factor;
    oskar_Image* image;

    if (*status) return;

    /* Return if a total intensity beam pattern has not been specified. */
    if (!(settings->beam_pattern.oskar_image_total_intensity ||
            settings->beam_pattern.fits_image_total_intensity))
        return;

    /* Dimensions of input beam pattern image to be converted to total intensity. */
    num_channels = oskar_image_num_channels(complex_cube);
    num_times = oskar_image_num_times(complex_cube);
    num_pols = oskar_image_num_pols(complex_cube);
    width = oskar_image_width(complex_cube);
    height = oskar_image_height(complex_cube);
    num_pixels = width * height;

    /* Allocate total intensity image cube to write into. */
    image = oskar_image_create(type, OSKAR_CPU, status);
    /* Set the beam pattern image cube */
    oskar_image_resize(image, width, height, 1, num_times, num_channels,
            status);
    oskar_image_set_type(image, (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED);
    oskar_image_set_coord_frame(image, settings->beam_pattern.coord_frame_type);
    oskar_image_set_grid_type(image, settings->beam_pattern.coord_grid_type);
    oskar_image_set_healpix_nside(image, settings->beam_pattern.nside);
    oskar_image_set_type(image, OSKAR_IMAGE_TYPE_BEAM_SCALAR);
    oskar_image_set_centre(image,
            settings->obs.phase_centre_lon_rad[0] * 180.0 / M_PI,
            settings->obs.phase_centre_lat_rad[0] * 180.0 / M_PI);
    oskar_image_set_fov(image, settings->beam_pattern.fov_deg[0],
            settings->beam_pattern.fov_deg[1]);
    oskar_image_set_freq(image, settings->obs.start_frequency_hz,
            settings->obs.frequency_inc_hz);
    oskar_image_set_time(image, settings->obs.start_mjd_utc,
            settings->obs.dt_dump_days * 86400.0);
    oskar_mem_append_raw(oskar_image_settings_path(image),
            settings->settings_path, OSKAR_CHAR, OSKAR_CPU,
            1 + strlen(settings->settings_path), status);

    /* For polarised beams Stokes I is 0.5 * (XX + YY) */
    /* For scalar beams total intensity is voltage squared. */
    factor = (num_pols == 4) ? 0.5 : 1.0;

    if (type == OSKAR_SINGLE)
    {
        float* image_data;
        const float2* complex_data;
        image_data = oskar_mem_float(oskar_image_data(image), status);
        complex_data = oskar_mem_float2_const(
                oskar_image_data_const(complex_cube), status);
        for (c = 0, idx = 0, islice = 0; c < num_channels; ++c)
        {
            for (t = 0; t < num_times; ++t, ++islice)
            {
                float* image_plane = &(image_data[islice * num_pixels]);
                for (p = 0; p < num_pols; ++p)
                {
                    for (i = 0; i < num_pixels; ++i, ++idx)
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
        double* image_data;
        const double2* complex_data;
        image_data = oskar_mem_double(oskar_image_data(image), status);
        complex_data = oskar_mem_double2_const(
                oskar_image_data_const(complex_cube), status);
        for (c = 0, idx = 0, islice = 0; c < num_channels; ++c)
        {
            for (t = 0; t < num_times; ++t, ++islice)
            {
                double* image_plane = &(image_data[islice * num_pixels]);
                for (p = 0; p < num_pols; ++p)
                {
                    for (i = 0; i < num_pixels; ++i, ++idx)
                    {
                        double xx = complex_data[idx].x * complex_data[idx].x;
                        double yy = complex_data[idx].y * complex_data[idx].y;
                        image_plane[i] += factor * (xx + yy);
                    }
                }
            }
        }
    }

    /* Write OSKAR image. */
    filename = settings->beam_pattern.oskar_image_total_intensity;
    if (filename && !*status)
    {
        oskar_log_message(log, 'M', 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(image, log, filename, 0, status);
    }

    /* Write FITS image. */
    filename = settings->beam_pattern.fits_image_total_intensity;
    if (filename && !*status)
    {
        oskar_log_message(log, 'M', 0, "Writing FITS image file: '%s'", filename);
        oskar_fits_image_write(image, log, filename, status);
    }

    oskar_image_free(image, status);
}

#ifdef __cplusplus
}
#endif
