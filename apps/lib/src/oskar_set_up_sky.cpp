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

#include "apps/lib/oskar_set_up_sky.h"
#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_to_sky_model.h"
#include "fits/oskar_fits_healpix_to_sky_model.h"
#endif
#include "math/oskar_healpix_nside_to_npix.h"
#include "math/oskar_healpix_pix_to_angles_ring.h"
#include "math/oskar_random_gaussian.h"
#include "math/oskar_random_power_law.h"
#include "math/oskar_random_broken_power_law.h"
#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "sky/oskar_sky_model_append_to_set.h"
#include "sky/oskar_sky_model_combine_set.h"
#include "sky/oskar_sky_model_compute_relative_lmn.h"
#include "sky/oskar_sky_model_filter_by_flux.h"
#include "sky/oskar_sky_model_filter_by_radius.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_load_gsm.h"
#include "sky/oskar_sky_model_read.h"
#include "sky/oskar_sky_model_save.h"
#include "sky/oskar_sky_model_set_gaussian_parameters.h"
#include "sky/oskar_sky_model_set_source.h"
#include "sky/oskar_sky_model_set_spectral_index.h"
#include "sky/oskar_sky_model_write.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_value.h"
#include "utility/oskar_log_warning.h"
#include "utility/oskar_get_error_string.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const int width = 45;

static int set_up_filter(oskar_SkyModel* sky,
        const oskar_SettingsSkyFilter* filter, double ra0_rad, double dec0_rad);
static int set_up_extended(oskar_SkyModel* sky,
        const oskar_SettingsSkyExtendedSources* ext, oskar_Log* log,
        double ra0_rad, double dec0_rad, int zero_failed_sources);

extern "C"
int oskar_set_up_sky(int* num_chunks, oskar_SkyModel** sky_chunks,
        oskar_Log* log, const oskar_Settings* settings)
{
    int error = 0;
    const char* filename;
    oskar_log_section(log, "Sky model");

    /* Sky model data type. */
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int max_sources_per_chunk = settings->sim.max_sources_per_chunk;

    /* Bool switch to set to zero the amplitude of sources where no Gaussian
     * source solution can be found. */
    int zero_failed_sources = OSKAR_FALSE;

    /* Load OSKAR sky files. */
    for (int i = 0; i < settings->sky.num_sky_files; ++i)
    {
        filename = settings->sky.input_sky_file[i];
        if (filename)
        {
            if (strlen(filename) > 0)
            {
                int binary_file_error = 0;

                /* Load into a temporary structure. */
                binary_file_error = 0;
                oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
                oskar_log_message(log, 0, "Loading source file '%s' ...",
                        filename);

                /* Try to read sky model as a binary file first. */
                /* If this fails, read it as an ASCII file. */
                oskar_sky_model_read(&temp, filename, OSKAR_LOCATION_CPU,
                        &binary_file_error);
                if (binary_file_error)
                    oskar_sky_model_load(&temp, filename, &error);
                if (error) return error;

                /* Apply filters and extended source over-ride. */
                error = set_up_filter(&temp, &settings->sky.input_sky_filter,
                        settings->obs.ra0_rad, settings->obs.dec0_rad);
                if (error) return error;
                error = set_up_extended(&temp,
                        &settings->sky.input_sky_extended_sources, log,
                        settings->obs.ra0_rad, settings->obs.dec0_rad,
                        zero_failed_sources);
                if (error) return error;

                /* Append to chunks. */
                error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                        max_sources_per_chunk, &temp);
                if (error) return error;

                oskar_log_message(log, 1, "done.");
            }
        }
    }

    /* GSM sky model file. */
    filename = settings->sky.gsm_file;
    if (filename)
    {
        if (strlen(filename) > 0)
        {
            /* Load the sky model data into a temporary sky model. */
            oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
            oskar_log_message(log, 0, "Loading GSM data...");
            error = oskar_sky_model_load_gsm(&temp, log, filename);
            if (error) return error;

            /* Apply filters and extended source over-ride. */
            error = set_up_filter(&temp, &settings->sky.gsm_filter,
                    settings->obs.ra0_rad, settings->obs.dec0_rad);
            if (error) return error;
            error = set_up_extended(&temp,
                    &settings->sky.gsm_extended_sources, log,
                    settings->obs.ra0_rad, settings->obs.dec0_rad,
                    zero_failed_sources);
            if (error) return error;

            /* Append to chunks. */
            error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                    max_sources_per_chunk, &temp);
            if (error) return error;

            oskar_log_message(log, 1, "done.");
        }
    }

#ifndef OSKAR_NO_FITS
    /* Load FITS image files. */
    for (int i = 0; i < settings->sky.num_fits_files; ++i)
    {
        filename = settings->sky.fits_file[i];
        if (filename)
        {
            if (strlen(filename) > 0)
            {
                /* Load into a temporary structure. */
                oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
                oskar_log_message(log, 0, "Loading FITS file '%s' ...",
                        filename);
                error = oskar_fits_to_sky_model(log, filename, &temp,
                        settings->sky.fits_file_settings.spectral_index,
                        settings->sky.fits_file_settings.min_peak_fraction,
                        settings->sky.fits_file_settings.noise_floor,
                        settings->sky.fits_file_settings.downsample_factor);
                if (error) return error;

                /* Append to chunks. */
                error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                        max_sources_per_chunk, &temp);
                if (error) return error;

                oskar_log_message(log, 1, "done.");
            }
        }
    }

    /* Load HEALPix FITS image files. */
    for (int i = 0; i < settings->sky.healpix_fits.num_files; ++i)
    {
        filename = settings->sky.healpix_fits.file[i];
        if (filename)
        {
            if (strlen(filename) > 0)
            {
                /* Load into a temporary structure. */
                oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
                oskar_log_message(log, 0, "Loading HEALPix FITS file '%s' ...",
                        filename);
                oskar_fits_healpix_to_sky_model(log, filename,
                        &settings->sky.healpix_fits, &temp, &error);
                if (error) return error;

                /* Apply filters and extended source over-ride. */
                error = set_up_filter(&temp, &settings->sky.healpix_fits.filter,
                        settings->obs.ra0_rad, settings->obs.dec0_rad);
                if (error) return error;
                error = set_up_extended(&temp,
                        &settings->sky.healpix_fits.extended_sources, log,
                        settings->obs.ra0_rad, settings->obs.dec0_rad,
                        zero_failed_sources);
                if (error) return error;

                /* Append to chunks. */
                error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                        max_sources_per_chunk, &temp);
                if (error) return error;

                oskar_log_message(log, 1, "done.");
            }
        }
    }
#endif

    /* HEALPix generator. */
    if (settings->sky.generator.healpix.nside != 0)
    {
        int nside, npix;

        /* Get the generator parameters. */
        nside = settings->sky.generator.healpix.nside;
        npix = oskar_healpix_nside_to_npix(nside);

        /* Generate the new positions into a temporary sky model. */
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, npix);
        oskar_log_message(log, 0, "Generating HEALPix source positions...");
        #pragma omp parallel for
        for (int i = 0; i < npix; ++i)
        {
            double ra, dec;
            oskar_healpix_pix_to_angles_ring(nside, i, &dec, &ra);
            dec = M_PI / 2.0 - dec;
            oskar_sky_model_set_source(&temp, i, ra, dec, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }
        if (error) return error;

        /* Apply filters and extended source over-ride. */
        error = set_up_filter(&temp, &settings->sky.generator.healpix.filter,
                settings->obs.ra0_rad, settings->obs.dec0_rad);
        if (error) return error;
        error = set_up_extended(&temp,
                &settings->sky.generator.healpix.extended_sources, log,
                settings->obs.ra0_rad, settings->obs.dec0_rad,
                zero_failed_sources);
        if (error) return error;

        /* Append to chunks. */
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }

    /* Random power-law generator. */
    if (settings->sky.generator.random_power_law.num_sources != 0)
    {
        const oskar_SettingsSkyGeneratorRandomPowerLaw* g =
                &settings->sky.generator.random_power_law;

        /* Generate the new positions into a temporary sky model. */
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, g->num_sources);
        srand(settings->sky.generator.random_power_law.seed);
        oskar_log_message(log, 0,
                "Generating random power law source distribution...");
        // Cannot parallelise here, since rand() is not thread safe.
        for (int i = 0; i < g->num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_power_law(g->flux_min, g->flux_max, g->power);
            oskar_sky_model_set_source(&temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }
        if (error) return error;

        /* Apply filters and extended source over-ride. */
        error = set_up_filter(&temp, &g->filter, settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;
        error = set_up_extended(&temp, &g->extended_sources, log,
                settings->obs.ra0_rad, settings->obs.dec0_rad,
                zero_failed_sources);
        if (error) return error;

        /* Append to chunks. */
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }

    /* Random broken power-law generator. */
    if (settings->sky.generator.random_broken_power_law.num_sources != 0)
    {
        const oskar_SettingsSkyGeneratorRandomBrokenPowerLaw* g =
                &settings->sky.generator.random_broken_power_law;

        /* Generate the new positions into a temporary sky model. */
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, g->num_sources);
        srand(settings->sky.generator.random_broken_power_law.seed);
        oskar_log_message(log, 0,
                "Generating random broken power law source distribution...");
        // Cannot parallelise here, since rand() is not thread safe.
        for (int i = 0; i < g->num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_broken_power_law(g->flux_min, g->flux_max,
                    g->threshold, g->power1, g->power2);
            oskar_sky_model_set_source(&temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }
        if (error) return error;

        /* Apply filters and extended source over-ride. */
        error = set_up_filter(&temp, &g->filter, settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;
        error = set_up_extended(&temp, &g->extended_sources, log,
                settings->obs.ra0_rad, settings->obs.dec0_rad,
                zero_failed_sources);
        if (error) return error;

        /* Append to chunks. */
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }

    /* Check if sky model contains no sources. */
    if (*num_chunks == 0)
    {
        oskar_log_warning(log, "Sky model contains no sources.");
    }
    else
    {
        /* Perform final pre-processing on chunk set. */
        int total_sources = 0, num_extended_chunks = 0;

        if (settings->sky.spectral_index.override)
        {
            double mean, std_dev, ref_freq, val;
            mean = settings->sky.spectral_index.mean;
            std_dev = settings->sky.spectral_index.std_dev;
            ref_freq = settings->sky.spectral_index.ref_frequency_hz;
            srand(settings->sky.spectral_index.seed);
            oskar_log_message(log, 0, "Overriding source spectral index values...");
            for (int i = 0; i < *num_chunks; ++i)
            {
                oskar_SkyModel* sky_chunk = &((*sky_chunks)[i]);

                /* Override source spectral index values. */
                for (int j = 0; j < sky_chunk->num_sources; ++j)
                {
                    val = oskar_random_gaussian(NULL) * std_dev + mean;
                    oskar_sky_model_set_spectral_index(sky_chunk, j,
                            ref_freq, val, &error);
                }
            }
            if (error) return error;
            oskar_log_message(log, 1, "done.");
        }

        oskar_log_message(log, 0, "Computing source direction cosines...");
        for (int i = 0; i < *num_chunks; ++i)
        {
            oskar_SkyModel* sky_chunk = &((*sky_chunks)[i]);

            /* Compute source direction cosines relative to phase centre. */
            oskar_sky_model_compute_relative_lmn(sky_chunk,
                    settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
            if (error) return error;

            /* Gather statistics on chunk set. */
            total_sources += sky_chunk->num_sources;
            if (sky_chunk->use_extended) ++num_extended_chunks;
        }
        oskar_log_message(log, 1, "done.");

        /* Print summary data. */
        oskar_log_message(log, 0, "Sky model summary");
        oskar_log_value(log, 1, width, "Num. sources", "%d", total_sources);
        oskar_log_value(log, 1, width, "Num. chunks", "%d", *num_chunks);
        oskar_log_value(log, 1, width, "Num. extended source chunks", "%d",
                num_extended_chunks);
#if (defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
        oskar_log_warning(log, "Extended sources disabled, as "
                "CBLAS and/or LAPACK were not found.");
#endif
    }

    /* Write out sky model if needed. */
    if (settings->sky.output_text_file || settings->sky.output_binary_file)
    {
        /* Concatenate chunks into a single sky model. */
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, 0);
        oskar_sky_model_combine_set(&temp, *sky_chunks, *num_chunks, &error);

        /* Write text file. */
        filename = settings->sky.output_text_file;
        if (filename)
        {
            if (strlen(filename))
            {
                oskar_log_message(log, 1, "Writing sky model text file: %s",
                        filename);
                oskar_sky_model_save(filename, &temp, &error);
            }
        }

        /* Write binary file. */
        filename = settings->sky.output_binary_file;
        if (filename)
        {
            if (strlen(filename))
            {
                oskar_log_message(log, 1, "Writing sky model binary file: %s",
                        filename);
                oskar_sky_model_write(filename, &temp, &error);
            }
        }
    }

    return error;
}


static int set_up_filter(oskar_SkyModel* sky,
        const oskar_SettingsSkyFilter* filter, double ra0_rad, double dec0_rad)
{
    int error = 0;
    double inner, outer, flux_min, flux_max;

    inner = filter->radius_inner;
    outer = filter->radius_outer;
    flux_min = filter->flux_min;
    flux_max = filter->flux_max;
    error = oskar_sky_model_filter_by_flux(sky, flux_min, flux_max);
    if (error) return error;
    error = oskar_sky_model_filter_by_radius(sky, inner, outer, ra0_rad,
            dec0_rad);
    return error;
}

static int set_up_extended(oskar_SkyModel* sky,
        const oskar_SettingsSkyExtendedSources* ext, oskar_Log* log,
        double ra0_rad, double dec0_rad, int zero_failed_sources)
{
    int error = 0;
    double FWHM_major, FWHM_minor, position_angle;

    FWHM_major = ext->FWHM_major;
    FWHM_minor = ext->FWHM_minor;
    position_angle = ext->position_angle;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
    /* Apply extended source over-ride. */
    if (FWHM_major > 0.0 || FWHM_minor > 0.0)
    {
        oskar_sky_model_set_gaussian_parameters(sky, FWHM_major,
                FWHM_minor, position_angle, &error);
    }
    if (error) return error;

    /* Evaluate extended source parameters. */
    /* FIXME Added sky->I as a hack to see if zeroing failed sources
     * makes a significant difference to simulations. */
    /* FIXME If this hack stays, must also zero Stokes Q, U, V as well. */
    error = oskar_evaluate_gaussian_source_parameters(log, sky->num_sources,
            &sky->gaussian_a, &sky->gaussian_b, &sky->gaussian_c,
            &sky->FWHM_major, &sky->FWHM_minor, &sky->position_angle,
            &sky->RA, &sky->Dec, zero_failed_sources, &sky->I, ra0_rad,
            dec0_rad);
#endif
    return error;
}

