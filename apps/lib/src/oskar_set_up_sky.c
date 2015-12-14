/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "apps/lib/oskar_set_up_sky.h"
#include <fits/oskar_fits_healpix_to_sky_model.h>
#include <fits/oskar_fits_image_to_sky_model.h>
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_random_gaussian.h>
#include <oskar_random_power_law.h>
#include <oskar_random_broken_power_law.h>
#include <oskar_generate_random_coordinate.h>
#include <oskar_sky.h>
#include <oskar_log.h>
#include <oskar_get_error_string.h>

#include <oskar_cmath.h>
#include <stdlib.h> /* For srand() */
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Suppress warnings about unused function arguments when LAPACK not found. */
#if defined(OSKAR_NO_LAPACK)
#   if defined(__INTEL_COMPILER)
#       pragma warning push
/*#       pragma warning(disable:XXX)*/
#   elif defined(__GNUC__)
#       pragma GCC diagnostic push
#       pragma GCC diagnostic ignored "-Wunused-parameter"
#   endif
#endif

static void set_up_osm(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyOskar* s, double ra0, double dec0,
        int zero_failed, int* status);
static void set_up_gsm(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGsm* s, double ra0, double dec0,
        int zero_failed, int* status);
static void set_up_fits_image(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyFitsImage* s, int* status);
static void set_up_healpix_fits(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyHealpixFits* s, double ra0, double dec0,
        int zero_failed, int* status);

static void set_up_gen_grid(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorGrid* s, double ra0, double dec0,
        int zero_failed, int* status);
static void set_up_gen_healpix(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorHealpix* s, double ra0, double dec0,
        int zero_failed, int* status);
static void set_up_gen_rpl(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomPowerLaw* s, double ra0,
        double dec0, int zero_failed, int* status);
static void set_up_gen_rbpl(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* s, double ra0,
        double dec0, int zero_failed, int* status);

static void write_sky_model(int num_chunks, oskar_Sky* const* chunks,
        const oskar_SettingsSky* s, oskar_Log* log, int* status);

static void set_up_filter(oskar_Sky* sky, const oskar_SettingsSkyFilter* f,
        double ra0_rad, double dec0_rad, int* status);
static void set_up_extended(oskar_Sky* sky,
        const oskar_SettingsSkyExtendedSources* ext, oskar_Log* log,
        double ra0_rad, double dec0_rad, int zero_failed_sources, int* status);
static void set_up_pol(oskar_Sky* sky,
        const oskar_SettingsSkyPolarisation* pol, int* status);


oskar_Sky** oskar_set_up_sky(const oskar_Settings* settings, oskar_Log* log,
        int* num_chunks, int* status)
{
    int max_per_chunk, type, zero_flag, i, j;
    int total_sources = 0, num_extended_chunks = 0;
    double ra0, dec0;
    oskar_Sky** sky_chunks = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Sky model data type and settings. */
    oskar_log_section(log, 'M', "Sky model");
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    max_per_chunk = settings->sim.max_sources_per_chunk;
    zero_flag = settings->sky.zero_failed_gaussians;
    ra0  = settings->obs.phase_centre_lon_rad[0];
    dec0 = settings->obs.phase_centre_lat_rad[0];

    /* Load sky model data files. */
    set_up_osm(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.oskar_sky_model, ra0, dec0, zero_flag, status);
    set_up_gsm(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.gsm, ra0, dec0, zero_flag, status);
    set_up_fits_image(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.fits_image, status);
    set_up_healpix_fits(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.healpix_fits, ra0, dec0, zero_flag, status);

    /* Generate sky models from generator parameters. */
    set_up_gen_grid(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.generator.grid, ra0, dec0, zero_flag, status);
    set_up_gen_healpix(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.generator.healpix, ra0, dec0, zero_flag, status);
    set_up_gen_rpl(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.generator.random_power_law, ra0, dec0, zero_flag,
            status);
    set_up_gen_rbpl(num_chunks, &sky_chunks, max_per_chunk, type, log,
            &settings->sky.generator.random_broken_power_law, ra0, dec0,
            zero_flag, status);

    /* Return if sky model contains no sources. */
    if (*num_chunks == 0)
    {
        oskar_log_warning(log, "Sky model contains no sources.");
        return sky_chunks;
    }

    /* Perform final pre-processing on chunk set. */
    if (settings->sky.spectral_index.override)
    {
        double mean, std_dev, ref_freq, val[2];
        mean = settings->sky.spectral_index.mean;
        std_dev = settings->sky.spectral_index.std_dev;
        ref_freq = settings->sky.spectral_index.ref_frequency_hz;
        oskar_log_message(log, 'M', 0, "Overriding source spectral index values...");
        for (i = 0; i < *num_chunks; ++i)
        {
            int num_sources;
            num_sources = oskar_sky_num_sources(sky_chunks[i]);

            /* Override source spectral index values. */
            for (j = 0; j < num_sources; ++j)
            {
                oskar_random_gaussian2(settings->sky.spectral_index.seed,
                        j, i, val);
                val[0] = std_dev * val[0] + mean;
                oskar_sky_set_spectral_index(sky_chunks[i], j,
                        ref_freq, val[0], status);
            }
        }
        oskar_log_message(log, 'M', 1, "done.");
    }

    if (*status) return sky_chunks;

    oskar_log_message(log, 'M', 0, "Computing source direction cosines...");
    for (i = 0; i < *num_chunks; ++i)
    {
        /* Compute source direction cosines relative to phase centre. */
        oskar_sky_evaluate_relative_directions(sky_chunks[i], ra0, dec0, status);

        /* Gather statistics on chunk set. */
        total_sources += oskar_sky_num_sources(sky_chunks[i]);
        if (oskar_sky_use_extended(sky_chunks[i]))
            ++num_extended_chunks;
    }
    oskar_log_message(log, 'M', 1, "done.");

    /* Print summary data. */
    oskar_log_message(log, 'M', 0, "Sky model summary");
    oskar_log_value(log, 'M', 1, "Num. sources", "%d", total_sources);
    oskar_log_value(log, 'M', 1, "Num. chunks", "%d", *num_chunks);
    oskar_log_value(log, 'M', 1, "Num. extended source chunks", "%d", num_extended_chunks);
#if defined(OSKAR_NO_LAPACK)
    oskar_log_warning(log, "Extended sources disabled (LAPACK not found).");
#endif

    /* Save final sky model set if required. */
    write_sky_model(*num_chunks, sky_chunks, &settings->sky, log, status);

    return sky_chunks;
}


static void set_up_osm(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyOskar* s, double ra0, double dec0,
        int zero_failed, int* status)
{
    int i;
    const char* filename;
    oskar_Sky* temp;

    /* Load OSKAR sky model files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        filename = s->file[i];
        if (filename && strlen(filename) > 0)
        {
            int binary_file_error = 0;

            /* Load into a temporary structure. */
            oskar_log_message(log, 'M', 0, "Loading OSKAR sky model file '%s' ...",
                    filename);

            /* Try to read sky model as a binary file first. */
            /* If this fails, read it as an ASCII file. */
            temp = oskar_sky_read(filename, OSKAR_CPU, &binary_file_error);
            if (binary_file_error)
                temp = oskar_sky_load(filename, type, status);

            /* Apply filters and extended source over-ride. */
            set_up_filter(temp, &s->filter, ra0, dec0, status);
            set_up_extended(temp, &s->extended_sources, log, ra0, dec0,
                    zero_failed, status);

            /* Append to chunks. */
            oskar_sky_append_to_set(num_chunks, sky_chunks,
                    max_per_chunk, temp, status);
            oskar_sky_free(temp, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
    }
}


static void set_up_gsm(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGsm* s, double ra0, double dec0,
        int zero_failed, int* status)
{
    const char* filename;
    oskar_Sky* temp;

    /* GSM sky model file. */
    if (*status) return;
    filename = s->file;
    if (filename && strlen(filename) > 0)
    {
        /* Load the sky model data into a temporary sky model. */
        temp = oskar_sky_create(type, OSKAR_CPU, 0, status);
        oskar_log_message(log, 'M', 0, "Loading GSM data...");
        oskar_sky_load_gsm(temp, filename, status);

        /* Apply filters and extended source over-ride. */
        set_up_filter(temp, &s->filter, ra0, dec0, status);
        set_up_extended(temp, &s->extended_sources, log, ra0, dec0,
                zero_failed, status);

        /* Append to chunks. */
        oskar_sky_append_to_set(num_chunks, sky_chunks,
                max_per_chunk, temp, status);
        oskar_sky_free(temp, status);
        oskar_log_message(log, 'M', 1, "done.");
    }
}


static void set_up_fits_image(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyFitsImage* s, int* status)
{
    int i;
    const char* filename;
    oskar_Sky* temp;

    /* Load FITS image files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        filename = s->file[i];
        if (filename && strlen(filename) > 0)
        {
            /* Load into a temporary structure. */
            temp = oskar_sky_create(type, OSKAR_CPU, 0, status);
            oskar_log_message(log, 'M', 0, "Loading FITS file '%s' ...",
                    filename);
            *status = oskar_fits_image_to_sky_model(log, filename, temp,
                    s->spectral_index, s->min_peak_fraction, s->noise_floor,
                    s->downsample_factor);
            if (*status)
            {
                oskar_sky_free(temp, status);
                return;
            }

            /* Append to chunks. */
            oskar_sky_append_to_set(num_chunks, sky_chunks,
                    max_per_chunk, temp, status);
            oskar_sky_free(temp, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
    }
}


static void set_up_healpix_fits(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyHealpixFits* s, double ra0, double dec0,
        int zero_failed, int* status)
{
    int i;
    const char* filename;
    oskar_Sky* temp;

    /* Load HEALPix FITS image files. */
    if (*status) return;
    for (i = 0; i < s->num_files; ++i)
    {
        filename = s->file[i];
        if (filename && strlen(filename) > 0)
        {
            /* Load into a temporary structure. */
            temp = oskar_sky_create(type, OSKAR_CPU, 0, status);
            oskar_log_message(log, 'M', 0, "Loading HEALPix FITS file '%s' ...",
                    filename);
            oskar_fits_healpix_to_sky_model(log, filename, s, temp, status);

            /* Apply filters and extended source over-ride. */
            set_up_filter(temp, &s->filter, ra0, dec0, status);
            set_up_extended(temp, &s->extended_sources, log, ra0, dec0,
                    zero_failed, status);

            /* Append to chunks. */
            oskar_sky_append_to_set(num_chunks, sky_chunks,
                    max_per_chunk, temp, status);
            oskar_sky_free(temp, status);
            oskar_log_message(log, 'M', 1, "done.");
        }
    }
}


static void set_up_gen_grid(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorGrid* s, double ra0, double dec0,
        int zero_failed, int* status)
{
    int i, j, k, num_points, side_length;
    double fov_rad, mean_flux, std_flux, r[2];
    oskar_Sky* temp;

    /* Get the grid generator parameters. */
    side_length = s->side_length;
    fov_rad = s->fov_rad;
    mean_flux = s->mean_flux_jy;
    std_flux = s->std_flux_jy;
    if (*status || side_length <= 0)
        return;

    /* Create a temporary sky model. */
    num_points = side_length * side_length;
    temp = oskar_sky_create(type, OSKAR_CPU, num_points, status);
    oskar_log_message(log, 'M', 0, "Generating source grid positions...");

    /* Side length of 1 is a special case. */
    if (side_length == 1)
    {
        /* Generate the Stokes I flux and store the value. */
        oskar_random_gaussian2(s->seed, 0, 0, r);
        r[0] = mean_flux + std_flux * r[0];
        oskar_sky_set_source(temp, 0, ra0, dec0, r[0], 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }
    else
    {
        double l_max, l, m, n, sin_dec0, cos_dec0, ra, dec;
        l_max = sin(0.5 * fov_rad);
        sin_dec0 = sin(dec0);
        cos_dec0 = cos(dec0);
        for (j = 0, k = 0; j < side_length; ++j)
        {
            m = 2.0 * l_max * j / (side_length - 1) - l_max;
            for (i = 0; i < side_length; ++i, ++k)
            {
                l = -2.0 * l_max * i / (side_length - 1) + l_max;

                /* Get longitude and latitude from tangent plane coords. */
                n = sqrt(1.0 - l*l - m*m);
                dec = asin(n * sin_dec0 + m * cos_dec0);
                ra = ra0 + atan2(l, cos_dec0 * n - m * sin_dec0);

                /* Generate the Stokes I flux and store the value. */
                oskar_random_gaussian2(s->seed, i, j, r);
                r[0] = mean_flux + std_flux * r[0];
                oskar_sky_set_source(temp, k, ra, dec, r[0], 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
            }
        }
    }

    /* Apply polarisation and extended source over-ride. */
    set_up_pol(temp, &s->pol, status);
    set_up_extended(temp, &s->extended_sources, log, ra0, dec0, zero_failed,
            status);

    /* Append to chunks. */
    oskar_sky_append_to_set(num_chunks, sky_chunks, max_per_chunk, temp,
            status);
    oskar_sky_free(temp, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_healpix(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorHealpix* s, double ra0, double dec0,
        int zero_failed, int* status)
{
    int i, nside, npix;
    oskar_Sky* temp;

    /* Get the HEALPix generator parameters. */
    nside = s->nside;
    if (*status || nside <= 0)
        return;

    /* Generate the new positions into a temporary sky model. */
    npix = 12 * nside * nside;
    temp = oskar_sky_create(type, OSKAR_CPU, npix, status);
    oskar_log_message(log, 'M', 0, "Generating HEALPix source positions...");
#pragma omp parallel for private(i)
    for (i = 0; i < npix; ++i)
    {
        double ra, dec;
        oskar_convert_healpix_ring_to_theta_phi_d(nside, i, &dec, &ra);
        dec = M_PI / 2.0 - dec;
        oskar_sky_set_source(temp, i, ra, dec, s->amplitude, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(temp, &s->filter, ra0, dec0, status);
    set_up_extended(temp, &s->extended_sources, log, ra0, dec0, zero_failed,
            status);

    /* Append to chunks. */
    oskar_sky_append_to_set(num_chunks, sky_chunks, max_per_chunk, temp,
            status);
    oskar_sky_free(temp, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_rpl(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomPowerLaw* s, double ra0,
        double dec0, int zero_failed, int* status)
{
    int i, num_sources;
    oskar_Sky* temp;

    /* Random power-law generator. */
    num_sources = s->num_sources;
    if (*status || num_sources <= 0)
        return;

    /* Generate the new positions into a temporary sky model. */
    temp = oskar_sky_create(type, OSKAR_CPU, num_sources, status);
    oskar_log_message(log, 'M', 0, "Generating random power law source distribution...");

    /* Cannot parallelise here, since rand() is not thread safe. */
    srand(s->seed);
    for (i = 0; i < num_sources; ++i)
    {
        double ra, dec, b;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_power_law(s->flux_min, s->flux_max, s->power);
        oskar_sky_set_source(temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(temp, &s->filter, ra0, dec0, status);
    set_up_extended(temp, &s->extended_sources, log, ra0, dec0, zero_failed,
            status);

    /* Append to chunks. */
    oskar_sky_append_to_set(num_chunks, sky_chunks, max_per_chunk, temp,
            status);
    oskar_sky_free(temp, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void set_up_gen_rbpl(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* s, double ra0,
        double dec0, int zero_failed, int* status)
{
    int i, num_sources;
    oskar_Sky* temp;

    /* Random broken power-law generator. */
    num_sources = s->num_sources;
    if (*status || num_sources <= 0)
        return;

    /* Generate the new positions into a temporary sky model. */
    temp = oskar_sky_create(type, OSKAR_CPU, num_sources, status);
    oskar_log_message(log, 'M', 0, "Generating random broken power law source distribution...");

    /* Cannot parallelise here, since rand() is not thread safe. */
    srand(s->seed);
    for (i = 0; i < num_sources; ++i)
    {
        double ra, dec, b;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_broken_power_law(s->flux_min, s->flux_max,
                s->threshold, s->power1, s->power2);
        oskar_sky_set_source(temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    /* Apply filters and extended source over-ride. */
    set_up_filter(temp, &s->filter, ra0, dec0, status);
    set_up_extended(temp, &s->extended_sources, log, ra0, dec0, zero_failed,
            status);

    /* Append to chunks. */
    oskar_sky_append_to_set(num_chunks, sky_chunks, max_per_chunk, temp,
            status);
    oskar_sky_free(temp, status);
    oskar_log_message(log, 'M', 1, "done.");
}


static void write_sky_model(int num_chunks, oskar_Sky* const* chunks,
        const oskar_SettingsSky* s, oskar_Log* log, int* status)
{
    const char* filename;
    oskar_Sky* temp;

    /* Write out sky model if needed. */
    if (*status) return;
    if (!s->output_binary_file && !s->output_text_file)
        return;

    /* Concatenate chunks into a single sky model. */
    temp = oskar_sky_combine_set(chunks, num_chunks, status);

    /* Write text file. */
    filename = s->output_text_file;
    if (filename && strlen(filename))
    {
        oskar_log_message(log, 'M', 1, "Writing sky model text file: %s", filename);
        oskar_sky_save(filename, temp, status);
    }

    /* Write binary file. */
    filename = s->output_binary_file;
    if (filename && strlen(filename))
    {
        oskar_log_message(log, 'M', 1, "Writing sky model binary file: %s", filename);
        oskar_sky_write(filename, temp, status);
    }

    /* Free memory. */
    oskar_sky_free(temp, status);
}


static void set_up_filter(oskar_Sky* sky, const oskar_SettingsSkyFilter* f,
        double ra0_rad, double dec0_rad, int* status)
{
    oskar_sky_filter_by_flux(sky, f->flux_min, f->flux_max, status);
    oskar_sky_filter_by_radius(sky, f->radius_inner_rad, f->radius_outer_rad,
            ra0_rad, dec0_rad, status);
}


static void set_up_extended(oskar_Sky* sky,
        const oskar_SettingsSkyExtendedSources* ext, oskar_Log* log,
        double ra0_rad, double dec0_rad, int zero_failed_sources, int* status)
{
#if !defined(OSKAR_NO_LAPACK)
    int num_failed = 0;

    /* Apply extended source over-ride. */
    if (ext->FWHM_major_rad > 0.0 || ext->FWHM_minor_rad > 0.0)
    {
        oskar_sky_set_gaussian_parameters(sky, ext->FWHM_major_rad,
                ext->FWHM_minor_rad, ext->position_angle_rad, status);
    }

    /* Evaluate extended source parameters. */
    oskar_sky_evaluate_gaussian_source_parameters(sky,
            zero_failed_sources, ra0_rad, dec0_rad, &num_failed, status);

    if (num_failed > 0)
    {
        if (zero_failed_sources)
        {
            oskar_log_warning(log, "Gaussian ellipse solution failed for %i "
                    "sources. These sources will have their flux values set "
                    "to zero.",  num_failed);
        }
        else
        {
            oskar_log_warning(log, "Gaussian ellipse solution failed for %i "
                    "sources. These sources will be simulated as point objects.",
                    num_failed);
        }
    }
#endif
}


static void set_up_pol(oskar_Sky* sky,
        const oskar_SettingsSkyPolarisation* pol, int* status)
{
    oskar_sky_override_polarisation(sky, pol->mean_pol_fraction,
            pol->std_pol_fraction, pol->mean_pol_angle_rad,
            pol->std_pol_angle_rad, pol->seed, status);
}

#if defined(OSKAR_NO_LAPACK)
#   if defined(__INTEL_COMPILER)
#       pragma warning pop
#   elif defined(__GNUC__)
#       pragma GCC diagnostic pop
#   endif
#endif

#ifdef __cplusplus
}
#endif
