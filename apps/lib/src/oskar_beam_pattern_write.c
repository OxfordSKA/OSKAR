/*
 * Copyright (c) 2013, The University of Oxford
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
#include <oskar_element.h>
#include <oskar_image_init.h>
#include <oskar_image_resize.h>
#include <oskar_image_write.h>
#include <oskar_image_free.h>
#include <fits/oskar_fits_image_write.h>
#include <oskar_mem.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================== */
static void save_complex_(const oskar_Image* complex_cube,
        const oskar_Settings* settings, oskar_Log* log, int* status);
static void save_power_(oskar_Image* image_cube,
        const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log,
        int num_pixels_total, int* err);
static void save_phase_(const oskar_Image* complex_cube,
        oskar_Image* image_cube, const oskar_Settings* settings, int type,
        oskar_Log* log, int num_pixels_total, int* err);
static void save_total_intensity_(const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log, int* status);
/* ========================================================================== */

void oskar_beam_pattern_write(const oskar_Image* complex_cube,
        oskar_Settings* settings, int type, oskar_Log* log, int* status)
{
    /* Set up image cube for beam pattern output images. */
    int num_times, num_channels, num_pols, num_pixels, num_pixels_total;
    const int* size;
    oskar_Image image;

    size = settings->beam_pattern.size;
    num_times = settings->obs.num_time_steps;
    num_channels = settings->obs.num_channels;
    num_pols = settings->telescope.aperture_array.element_pattern.functional_type ==
            OSKAR_ELEMENT_TYPE_ISOTROPIC ? 1 : 4;
    num_pixels = size[0] * size[1];
    num_pixels_total = num_pixels * num_times * num_channels * num_pols;

    oskar_image_init(&image, type, OSKAR_LOCATION_CPU, status);

    /* Set the beam pattern image cube */
    oskar_image_resize(&image, size[0], size[1], num_pols, num_times,
            num_channels, status);
    image.image_type         = (num_pols == 1) ?
            OSKAR_IMAGE_TYPE_BEAM_SCALAR : OSKAR_IMAGE_TYPE_BEAM_POLARISED;
    image.centre_ra_deg      = settings->obs.ra0_rad[0] * 180.0 / M_PI;
    image.centre_dec_deg     = settings->obs.dec0_rad[0] * 180.0 / M_PI;
    image.fov_ra_deg         = settings->beam_pattern.fov_deg[0];
    image.fov_dec_deg        = settings->beam_pattern.fov_deg[1];
    image.freq_start_hz      = settings->obs.start_frequency_hz;
    image.freq_inc_hz        = settings->obs.frequency_inc_hz;
    image.time_inc_sec       = settings->obs.dt_dump_days * 86400.0;
    image.time_start_mjd_utc = settings->obs.start_mjd_utc;
    oskar_mem_copy(&image.settings_path, &settings->settings_path, status);

    /* Save the complex beam pattern. */
    save_complex_(complex_cube, settings, log, status);

    /* Save the power beam pattern. */
    save_power_(&image, complex_cube, settings, type, log, num_pixels_total,
            status);

    /* Save the phase beam pattern. */
    save_phase_(complex_cube, &image, settings, type, log, num_pixels_total,
            status);

    /* Save the total intensity beam pattern. */
    save_total_intensity_(complex_cube, settings, type, log, status);

    oskar_image_free(&image, status);
}

static void save_complex_(const oskar_Image* complex_cube,
        const oskar_Settings* settings, oskar_Log* log, int* status)
{
    const char* filename = settings->beam_pattern.oskar_image_complex;

    /* Return if there is an error or the filename has not been set. */
    if ((status && *status != OSKAR_SUCCESS) || !filename)
        return;
    oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
    oskar_image_write(complex_cube, log, filename, 0, status);
}

static void save_power_(oskar_Image* image_cube,
        const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log,
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
            image_data = oskar_mem_float(&image_cube->data, status);
            complex_data = oskar_mem_float2_const(&complex_cube->data, status);
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
            image_data = oskar_mem_double(&image_cube->data, status);
            complex_data = oskar_mem_double2_const(&complex_cube->data, status);
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
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(image_cube, log, filename, 0, status);
        }
#ifndef OSKAR_NO_FITS
        /* Write FITS image. */
        filename = settings->beam_pattern.fits_image_voltage;
        if (filename && !*status)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(image_cube, log, filename, status);
        }
#endif
    }
}

static void save_phase_(const oskar_Image* complex_cube,
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
            image_data = oskar_mem_float(&image_cube->data, status);
            complex_data = oskar_mem_float2_const(&complex_cube->data, status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }
        else if (type == OSKAR_DOUBLE)
        {
            double* image_data;
            const double2* complex_data;
            image_data = oskar_mem_double(&image_cube->data, status);
            complex_data = oskar_mem_double2_const(&complex_cube->data, status);
            for (i = 0; i < num_pixels_total; ++i)
            {
                image_data[i] = atan2(complex_data[i].y, complex_data[i].x);
            }
        }

        /* Write OSKAR image. */
        filename = settings->beam_pattern.oskar_image_phase;
        if (filename && !*status)
        {
            oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
            oskar_image_write(image_cube, log, filename, 0, status);
        }
#ifndef OSKAR_NO_FITS
        /* Write FITS image. */
        filename = settings->beam_pattern.fits_image_phase;
        if (filename && !*status)
        {
            oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
            oskar_fits_image_write(image_cube, log, filename, status);
        }
#endif
    }
}

static void save_total_intensity_(const oskar_Image* complex_cube,
        const oskar_Settings* settings, int type, oskar_Log* log, int* status)
{
    const char* filename;
    int num_channels, num_times, num_pols, num_pixels;
    const int* size;
    int c, t, p, i, idx, islice;
    double factor;
    oskar_Image image;

    if (*status) return;

    /* Return if a total intensity beam pattern has not been specified. */
    if (!(settings->beam_pattern.oskar_image_total_intensity ||
            settings->beam_pattern.fits_image_total_intensity))
        return;

    /* Dimensions of input beam pattern image to be converted to total intensity. */
    num_channels = complex_cube->num_channels;
    num_times = complex_cube->num_times;
    num_pols = complex_cube->num_pols;
    num_pixels = complex_cube->width * complex_cube->height;
    size = settings->beam_pattern.size;

    /* Allocate total intensity image cube to write into. */
    oskar_image_init(&image, type, OSKAR_LOCATION_CPU, status);
    /* Set the beam pattern image cube */
    oskar_image_resize(&image, size[0], size[1], 1, num_times,
            num_channels, status);
    image.image_type         = OSKAR_IMAGE_TYPE_BEAM_SCALAR;
    image.centre_ra_deg      = settings->obs.ra0_rad[0] * 180.0 / M_PI;
    image.centre_dec_deg     = settings->obs.dec0_rad[0] * 180.0 / M_PI;
    image.fov_ra_deg         = settings->beam_pattern.fov_deg[0];
    image.fov_dec_deg        = settings->beam_pattern.fov_deg[1];
    image.freq_start_hz      = settings->obs.start_frequency_hz;
    image.freq_inc_hz        = settings->obs.frequency_inc_hz;
    image.time_inc_sec       = settings->obs.dt_dump_days * 86400.0;
    image.time_start_mjd_utc = settings->obs.start_mjd_utc;
    oskar_mem_copy(&image.settings_path, &settings->settings_path, status);

    /* For polarised beams Stokes I is 0.5 * (XX + YY) */
    /* For scalar beams total intensity is voltage squared. */
    factor = (num_pols == 4) ? 0.5 : 1.0;

    if (type == OSKAR_SINGLE)
    {
        float* image_data;
        const float2* complex_data;
        complex_data = oskar_mem_float2_const(&complex_cube->data, status);
        image_data = oskar_mem_float(&image.data, status);
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
        image_data = oskar_mem_double(&image.data, status);
        complex_data = oskar_mem_double2_const(&complex_cube->data,
                status);
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
        oskar_log_message(log, 0, "Writing OSKAR image file: '%s'", filename);
        oskar_image_write(&image, log, filename, 0, status);
    }
#ifndef OSKAR_NO_FITS
    /* Write FITS image. */
    filename = settings->beam_pattern.fits_image_total_intensity;
    if (filename && !*status)
    {
        oskar_log_message(log, 0, "Writing FITS image file: '%s'", filename);
        oskar_fits_image_write(&image, log, filename, status);
    }
#endif

    oskar_image_free(&image, status);
}

#ifdef __cplusplus
}
#endif
