/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#ifndef OSKAR_NO_FITS
#   include <fits/oskar_fits_healpix_to_sky_model.h>
#   include <fits/oskar_fits_image_to_sky_model.h>
#endif
#include <oskar_healpix_nside_to_npix.h>
#include <oskar_convert_healpix_ring_to_theta_phi.h>
#include <oskar_random_gaussian.h>
#include <oskar_random_power_law.h>
#include <oskar_random_broken_power_law.h>
#include <oskar_generate_random_coordinate.h>
#include <oskar_sky.h>
#include <oskar_log.h>
#include <oskar_get_error_string.h>

#include <math.h>
#include <stdlib.h> /* For srand() */
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

static const int width = 45;

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


oskar_Sky** oskar_set_up_sky(int* num_chunks, oskar_Log* log,
        const oskar_Settings* settings, int* status)
{
    int max_per_chunk, type, zero_flag, i, j;
    int total_sources = 0, num_extended_chunks = 0;
    double ra0, dec0;
    oskar_Sky** sky_chunks = 0;

    /* Check all inputs. */
    if  (!num_chunks || !settings || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Sky model data type and settings. */
    oskar_log_section(log, "Sky model");
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    max_per_chunk = settings->sim.max_sources_per_chunk;
    zero_flag = settings->sky.zero_failed_gaussians;
    ra0  = settings->obs.ra0_rad[0];
    dec0 = settings->obs.dec0_rad[0];

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
        double mean, std_dev, ref_freq, val;
        mean = settings->sky.spectral_index.mean;
        std_dev = settings->sky.spectral_index.std_dev;
        ref_freq = settings->sky.spectral_index.ref_frequency_hz;
        srand(settings->sky.spectral_index.seed);
        oskar_log_message(log, 0, "Overriding source spectral index values...");
        for (i = 0; i < *num_chunks; ++i)
        {
            int num_sources;
            num_sources = oskar_sky_num_sources(sky_chunks[i]);

            /* Override source spectral index values. */
            for (j = 0; j < num_sources; ++j)
            {
                val = oskar_random_gaussian(0) * std_dev + mean;
                oskar_sky_set_spectral_index(sky_chunks[i], j,
                        ref_freq, val, status);
            }
        }
        oskar_log_message(log, 1, "done.");
    }

    if (*status) return sky_chunks;

    oskar_log_message(log, 0, "Computing source direction cosines...");
    for (i = 0; i < *num_chunks; ++i)
    {
        /* Compute source direction cosines relative to phase centre. */
        oskar_sky_compute_relative_lmn(sky_chunks[i], ra0, dec0, status);
        oskar_sky_compute_source_radius(sky_chunks[i], ra0, dec0, status);

        /* Gather statistics on chunk set. */
        total_sources += oskar_sky_num_sources(sky_chunks[i]);
        if (oskar_sky_use_extended(sky_chunks[i]))
            ++num_extended_chunks;
    }
    oskar_log_message(log, 1, "done.");

    /* Print summary data. */
    oskar_log_message(log, 0, "Sky model summary");
    oskar_log_value(log, 1, width, "Num. sources", "%d", total_sources);
    oskar_log_value(log, 1, width, "Num. chunks", "%d", *num_chunks);
    oskar_log_value(log, 1, width, "Num. extended source chunks", "%d",
            num_extended_chunks);
#if (defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    oskar_log_warning(log, "Extended sources disabled, as CBLAS and/or LAPACK "
            "were not found.");
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
            temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, 0, status);
            oskar_log_message(log, 0, "Loading OSKAR sky model file '%s' ...",
                    filename);

            /* Try to read sky model as a binary file first. */
            /* If this fails, read it as an ASCII file. */
            oskar_sky_read(temp, filename, OSKAR_LOCATION_CPU,
                    &binary_file_error);
            if (binary_file_error)
                oskar_sky_load(temp, filename, status);

            /* Apply filters and extended source over-ride. */
            set_up_filter(temp, &s->filter, ra0, dec0, status);
            set_up_extended(temp, &s->extended_sources, log, ra0, dec0,
                    zero_failed, status);

            /* Append to chunks. */
            oskar_sky_append_to_set(num_chunks, sky_chunks,
                    max_per_chunk, temp, status);
            oskar_sky_free(temp, status);
            oskar_log_message(log, 1, "done.");
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
        temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, 0, status);
        oskar_log_message(log, 0, "Loading GSM data...");
        oskar_sky_load_gsm(temp, filename, status);

        /* Apply filters and extended source over-ride. */
        set_up_filter(temp, &s->filter, ra0, dec0, status);
        set_up_extended(temp, &s->extended_sources, log, ra0, dec0,
                zero_failed, status);

        /* Append to chunks. */
        oskar_sky_append_to_set(num_chunks, sky_chunks,
                max_per_chunk, temp, status);
        oskar_sky_free(temp, status);
        oskar_log_message(log, 1, "done.");
    }
}


static void set_up_fits_image(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyFitsImage* s, int* status)
{
#ifndef OSKAR_NO_FITS
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
            temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, 0, status);
            oskar_log_message(log, 0, "Loading FITS file '%s' ...",
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
            oskar_log_message(log, 1, "done.");
        }
    }
#endif
}


static void set_up_healpix_fits(int* num_chunks, oskar_Sky*** sky_chunks,
        int max_per_chunk, int type, oskar_Log* log,
        const oskar_SettingsSkyHealpixFits* s, double ra0, double dec0,
        int zero_failed, int* status)
{
#ifndef OSKAR_NO_FITS
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
            temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, 0, status);
            oskar_log_message(log, 0, "Loading HEALPix FITS file '%s' ...",
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
            oskar_log_message(log, 1, "done.");
        }
    }
#endif
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
    npix = oskar_healpix_nside_to_npix(nside);
    temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, npix, status);
    oskar_log_message(log, 0, "Generating HEALPix source positions...");
#pragma omp parallel for private(i)
    for (i = 0; i < npix; ++i)
    {
        double ra, dec;
        oskar_convert_healpix_ring_to_theta_phi(nside, i, &dec, &ra);
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
    oskar_log_message(log, 1, "done.");
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
    temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, num_sources,
            status);
    oskar_log_message(log, 0,
            "Generating random power law source distribution...");

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
    oskar_log_message(log, 1, "done.");
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
    temp = oskar_sky_create(type, OSKAR_LOCATION_CPU, num_sources,
            status);
    oskar_log_message(log, 0,
            "Generating random broken power law source distribution...");

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
    oskar_log_message(log, 1, "done.");
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
        oskar_log_message(log, 1, "Writing sky model text file: %s", filename);
        oskar_sky_save(filename, temp, status);
    }

    /* Write binary file. */
    filename = s->output_binary_file;
    if (filename && strlen(filename))
    {
        oskar_log_message(log, 1, "Writing sky model binary file: %s", filename);
        oskar_sky_write(filename, temp, status);
    }

    /* Free memory. */
    oskar_sky_free(temp, status);
}


static void set_up_filter(oskar_Sky* sky, const oskar_SettingsSkyFilter* f,
        double ra0_rad, double dec0_rad, int* status)
{
    oskar_sky_filter_by_flux(sky, f->flux_min, f->flux_max, status);
    oskar_sky_filter_by_radius(sky, f->radius_inner, f->radius_outer,
            ra0_rad, dec0_rad, status);
}


static void set_up_extended(oskar_Sky* sky,
        const oskar_SettingsSkyExtendedSources* ext, oskar_Log* log,
        double ra0_rad, double dec0_rad, int zero_failed_sources, int* status)
{
#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    int num_failed = 0;

    /* Apply extended source over-ride. */
    if (ext->FWHM_major > 0.0 || ext->FWHM_minor > 0.0)
    {
        oskar_sky_set_gaussian_parameters(sky, ext->FWHM_major,
                ext->FWHM_minor, ext->position_angle, status);
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

#ifdef __cplusplus
}
#endif
