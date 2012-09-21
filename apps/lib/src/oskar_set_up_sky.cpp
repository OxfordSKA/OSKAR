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
#endif
#include "math/oskar_healpix_nside_to_npix.h"
#include "math/oskar_healpix_pix_to_angles_ring.h"
#include "math/oskar_random_power_law.h"
#include "math/oskar_random_broken_power_law.h"
#include "sky/oskar_evaluate_gaussian_source_parameters.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "sky/oskar_sky_model_append_to_set.h"
#include "sky/oskar_sky_model_combine_set.h"
#include "sky/oskar_sky_model_compute_relative_lmn.h"
#include "sky/oskar_sky_model_load.h"
#include "sky/oskar_sky_model_load_gsm.h"
#include "sky/oskar_sky_model_save.h"
#include "sky/oskar_sky_model_set_gaussian_parameters.h"
#include "sky/oskar_sky_model_set_source.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_section.h"
#include "utility/oskar_log_value.h"
#include "utility/oskar_log_warning.h"
#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const int width = 45;

extern "C"
int oskar_set_up_sky(int* num_chunks, oskar_SkyModel** sky_chunks,
        oskar_Log* log, const oskar_Settings* settings)
{
    int error = OSKAR_SUCCESS;
    const char* filename;
    oskar_log_section(log, "Sky model");

    // Sky model data type.
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int max_sources_per_chunk = settings->sim.max_sources_per_chunk;

    // Bool switch to set to zero the amplitude of sources where no gaussian
    // source solution can be found.
    int zero_failed_sources = OSKAR_FALSE;

    // Load OSKAR sky files.
    for (int i = 0; i < settings->sky.num_sky_files; ++i)
    {
        filename = settings->sky.input_sky_file[i];
        if (filename)
        {
            if (strlen(filename) > 0)
            {
                // Load into a temporary structure.
                oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
                oskar_log_message(log, 0, "Loading source file %s ...", filename);
                oskar_sky_model_load(&temp, filename, &error);
                if (error) return error;

                // Apply filters.
                double inner = settings->sky.input_sky_filter.radius_inner;
                double outer = settings->sky.input_sky_filter.radius_outer;
                double flux_min = settings->sky.input_sky_filter.flux_min;
                double flux_max = settings->sky.input_sky_filter.flux_max;
                error = temp.filter_by_flux(flux_min, flux_max);
                if (error) return error;
                error = temp.filter_by_radius(inner, outer, settings->obs.ra0_rad,
                        settings->obs.dec0_rad);
                if (error) return error;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
                // Apply extended source over-ride.
                double FWHM_major = settings->sky.input_sky_extended_sources.FWHM_major;
                double FWHM_minor = settings->sky.input_sky_extended_sources.FWHM_minor;
                double position_angle = settings->sky.input_sky_extended_sources.position_angle;
                if (FWHM_major > 0.0 || FWHM_minor > 0.0)
                {
                    error = oskar_sky_model_set_gaussian_parameters(&temp, FWHM_major,
                            FWHM_minor, position_angle);
                }
                if (error) return error;

                // Evaluate extended source parameters.
                error = oskar_evaluate_gaussian_source_parameters(log,
                        temp.num_sources, &temp.gaussian_a, &temp.gaussian_b,
                        &temp.gaussian_c, &temp.FWHM_major, &temp.FWHM_minor,
                        &temp.position_angle, &temp.RA, &temp.Dec,
                        zero_failed_sources, &temp.I,
                        settings->obs.ra0_rad, settings->obs.dec0_rad);
                if (error) return error;
#endif

                // Compute source direction cosines (relative lmn)
                oskar_sky_model_compute_relative_lmn(&temp,
                        settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
                if (error) return error;

                // Append to chunks.
                error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                        max_sources_per_chunk, &temp);
                if (error) return error;

                oskar_log_message(log, 1, "done.");
            }
        }
    }

    // GSM sky model file.
    filename = settings->sky.gsm_file;
    if (filename)
    {
        if (strlen(filename) > 0)
        {
            // Load the sky model data into a temporary sky model.
            oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
            oskar_log_message(log, 0, "Loading GSM data...");
            error = oskar_sky_model_load_gsm(&temp, log, filename);
            if (error) return error;

            // Apply filters.
            double inner = settings->sky.gsm_filter.radius_inner;
            double outer = settings->sky.gsm_filter.radius_outer;
            double flux_min = settings->sky.gsm_filter.flux_min;
            double flux_max = settings->sky.gsm_filter.flux_max;
            error = temp.filter_by_flux(flux_min, flux_max);
            if (error) return error;
            error = temp.filter_by_radius(inner, outer, settings->obs.ra0_rad,
                    settings->obs.dec0_rad);
            if (error) return error;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
            // Apply extended source over-ride.
            double FWHM_major = settings->sky.gsm_extended_sources.FWHM_major;
            double FWHM_minor = settings->sky.gsm_extended_sources.FWHM_minor;
            double position_angle = settings->sky.gsm_extended_sources.position_angle;
            if (FWHM_major > 0.0 || FWHM_minor > 0.0)
            {
                error = oskar_sky_model_set_gaussian_parameters(&temp, FWHM_major,
                        FWHM_minor, position_angle);
            }
            if (error) return error;

            // Evaluate extended source parameters.
            // FIXME added temp.I as a hack to see if zeroing failed sources
            // makes a significant difference to simulations
            error = oskar_evaluate_gaussian_source_parameters(log,
                    temp.num_sources, &temp.gaussian_a, &temp.gaussian_b,
                    &temp.gaussian_c, &temp.FWHM_major, &temp.FWHM_minor,
                    &temp.position_angle, &temp.RA, &temp.Dec,
                    zero_failed_sources, &temp.I, settings->obs.ra0_rad,
                    settings->obs.dec0_rad);
            if (error) return error;
#endif

            // Compute source direction cosines (relative lmn)
            oskar_sky_model_compute_relative_lmn(&temp,
                    settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
            if (error) return error;

            // Append to chunks.
            error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                    max_sources_per_chunk, &temp);
            if (error) return error;

            oskar_log_message(log, 1, "done.");
        }
    }

#ifndef OSKAR_NO_FITS
    // Load FITS image files.
    for (int i = 0; i < settings->sky.num_fits_files; ++i)
    {
        filename = settings->sky.fits_file[i];
        if (filename)
        {
            if (strlen(filename) > 0)
            {
                // Load into a temporary structure.
                oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
                oskar_log_message(log, 0, "Loading FITS file %s ...", filename);
                error = oskar_fits_to_sky_model(log, filename, &temp,
                        settings->sky.fits_file_settings.spectral_index,
                        settings->sky.fits_file_settings.min_peak_fraction,
                        settings->sky.fits_file_settings.noise_floor,
                        settings->sky.fits_file_settings.downsample_factor);
                if (error) return error;

                // Compute source direction cosines (relative lmn)
                oskar_sky_model_compute_relative_lmn(&temp,
                        settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
                if (error) return error;

                // Append to chunks.
                error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                        max_sources_per_chunk, &temp);
                if (error) return error;

                oskar_log_message(log, 1, "done.");
            }
        }
    }
#endif

    // HEALPix generator.
    if (settings->sky.generator.healpix.nside != 0)
    {
        // Get the generator parameters.
        int nside = settings->sky.generator.healpix.nside;
        int npix = oskar_healpix_nside_to_npix(nside);

        // Generate the new positions into a temporary sky model.
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, npix);
        oskar_log_message(log, 0, "Generating HEALPIX source positions...");
        #pragma omp parallel for
        for (int i = 0; i < npix; ++i)
        {
            double ra, dec;
            oskar_healpix_pix_to_angles_ring(nside, i, &dec, &ra);
            dec = M_PI / 2.0 - dec;
            oskar_sky_model_set_source(&temp, i, ra, dec, 1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }

        // Apply filters.
        double inner = settings->sky.generator.healpix.filter.radius_inner;
        double outer = settings->sky.generator.healpix.filter.radius_outer;
        double flux_min = settings->sky.generator.healpix.filter.flux_min;
        double flux_max = settings->sky.generator.healpix.filter.flux_max;
        error = temp.filter_by_flux(flux_min, flux_max);
        if (error) return error;
        error = temp.filter_by_radius(inner, outer, settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
        // Apply extended source over-ride.
        double FWHM_major = settings->sky.generator.healpix.extended_sources.FWHM_major;
        double FWHM_minor = settings->sky.generator.healpix.extended_sources.FWHM_minor;
        double position_angle = settings->sky.generator.healpix.extended_sources.position_angle;
        if (FWHM_major > 0.0 || FWHM_minor > 0.0)
        {
            error = oskar_sky_model_set_gaussian_parameters(&temp, FWHM_major,
                    FWHM_minor, position_angle);
        }
        if (error) return error;

        // Evaluate extended source parameters.
        error = oskar_evaluate_gaussian_source_parameters(log, temp.num_sources,
                &temp.gaussian_a, &temp.gaussian_b, &temp.gaussian_c,
                &temp.FWHM_major, &temp.FWHM_minor, &temp.position_angle,
                &temp.RA, &temp.Dec, zero_failed_sources, &temp.I,
                settings->obs.ra0_rad, settings->obs.dec0_rad);
        if (error) return error;
#endif

        // Compute source direction cosines (relative lmn)
        oskar_sky_model_compute_relative_lmn(&temp,
                settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
        if (error) return error;

        // Append to chunks.
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }

    // Random power-law generator.
    if (settings->sky.generator.random_power_law.num_sources != 0)
    {
        // Get the generator parameters.
        int num_sources = settings->sky.generator.random_power_law.num_sources;
        double min = settings->sky.generator.random_power_law.flux_min;
        double max = settings->sky.generator.random_power_law.flux_max;
        double power = settings->sky.generator.random_power_law.power;

        // Generate the new positions into a temporary sky model.
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, num_sources);
        srand(settings->sky.generator.random_power_law.seed);
        oskar_log_message(log, 0,
                "Generating random power law source distribution...");
        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_power_law(min, max, power);
            oskar_sky_model_set_source(&temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }

        // Apply filters.
        double inner = settings->sky.generator.random_power_law.filter.radius_inner;
        double outer = settings->sky.generator.random_power_law.filter.radius_outer;
        double flux_min = settings->sky.generator.random_power_law.filter.flux_min;
        double flux_max = settings->sky.generator.random_power_law.filter.flux_max;
        error = temp.filter_by_flux(flux_min, flux_max);
        if (error) return error;
        error = temp.filter_by_radius(inner, outer, settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
        // Apply extended source over-ride.
        double FWHM_major = settings->sky.generator.random_power_law.extended_sources.FWHM_major;
        double FWHM_minor = settings->sky.generator.random_power_law.extended_sources.FWHM_minor;
        double position_angle = settings->sky.generator.random_power_law.extended_sources.position_angle;
        if (FWHM_major > 0.0 || FWHM_minor > 0.0)
        {
            error = oskar_sky_model_set_gaussian_parameters(&temp, FWHM_major,
                    FWHM_minor, position_angle);
        }
        if (error) return error;

        // Evaluate extended source parameters.
        error = oskar_evaluate_gaussian_source_parameters(log, temp.num_sources,
                &temp.gaussian_a, &temp.gaussian_b, &temp.gaussian_c,
                &temp.FWHM_major, &temp.FWHM_minor, &temp.position_angle,
                &temp.RA, &temp.Dec, zero_failed_sources, &temp.I,
                settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;
#endif

        // Compute source direction cosines (relative lmn)
        oskar_sky_model_compute_relative_lmn(&temp,
                settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
        if (error) return error;

        // Append to chunks.
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }

    // Random broken power-law generator.
    if (settings->sky.generator.random_broken_power_law.num_sources != 0)
    {
        // Get the generator parameters.
        int num_sources = settings->sky.generator.random_broken_power_law.num_sources;
        double min = settings->sky.generator.random_broken_power_law.flux_min;
        double max = settings->sky.generator.random_broken_power_law.flux_max;
        double threshold = settings->sky.generator.random_broken_power_law.threshold;
        double power1 = settings->sky.generator.random_broken_power_law.power1;
        double power2 = settings->sky.generator.random_broken_power_law.power2;

        // Generate the new positions into a temporary sky model.
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, num_sources);
        srand(settings->sky.generator.random_broken_power_law.seed);
        oskar_log_message(log, 0,
                "Generating random broken power law source distribution...");
        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_broken_power_law(min, max, threshold, power1, power2);
            oskar_sky_model_set_source(&temp, i, ra, dec, b, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0, 0.0, &error);
        }

        // Apply filters.
        double inner = settings->sky.generator.random_broken_power_law.filter.radius_inner;
        double outer = settings->sky.generator.random_broken_power_law.filter.radius_outer;
        double flux_min = settings->sky.generator.random_broken_power_law.filter.flux_min;
        double flux_max = settings->sky.generator.random_broken_power_law.filter.flux_max;
        error = temp.filter_by_flux(flux_min, flux_max);
        if (error) return error;
        error = temp.filter_by_radius(inner, outer, settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        if (error) return error;

#if !(defined(OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK))
        // Apply extended source over-ride.
        double FWHM_major = settings->sky.generator.random_broken_power_law.extended_sources.FWHM_major;
        double FWHM_minor = settings->sky.generator.random_broken_power_law.extended_sources.FWHM_minor;
        double position_angle = settings->sky.generator.random_broken_power_law.extended_sources.position_angle;
        if (FWHM_major > 0.0 || FWHM_minor > 0.0)
        {
            error = oskar_sky_model_set_gaussian_parameters(&temp, FWHM_major,
                    FWHM_minor, position_angle);
        }
        if (error) return error;

        // Evaluate extended source parameters.
        error = oskar_evaluate_gaussian_source_parameters(log, temp.num_sources,
                &temp.gaussian_a, &temp.gaussian_b, &temp.gaussian_c,
                &temp.FWHM_major, &temp.FWHM_minor, &temp.position_angle,
                &temp.RA, &temp.Dec, zero_failed_sources, &temp.I,
                settings->obs.ra0_rad, settings->obs.dec0_rad);
        if (error) return error;
#endif

        // Compute source direction cosines (relative lmn)
        oskar_sky_model_compute_relative_lmn(&temp,
                settings->obs.ra0_rad, settings->obs.dec0_rad, &error);
        if (error) return error;

        // Append to chunks.
        error = oskar_sky_model_append_to_set(num_chunks, sky_chunks,
                max_sources_per_chunk, &temp);
        if (error) return error;

        oskar_log_message(log, 1, "done.");
    }


    // Check if sky model contains no sources.
    if (*num_chunks == 0)
        oskar_log_warning(log, "Sky model contains no sources.");
    else
    {
        // Print summary data.
        int total_sources = (*num_chunks - 1) * max_sources_per_chunk + ((*sky_chunks)[*num_chunks-1]).num_sources;
        int num_extended_chunks = 0;
        for (int i = 0; i < *num_chunks; ++i)
        {
            if (((*sky_chunks)[i]).use_extended) ++num_extended_chunks;
        }

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

    // Write sky model out.
    filename = settings->sky.output_sky_file;
    if (filename)
    {
        if (strlen(filename))
        {
            oskar_log_message(log, 1, "Writing sky model file to disk as: %s",
                    filename);
            oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, 0);
            oskar_sky_model_combine_set(&temp, *sky_chunks, *num_chunks, &error);
            oskar_sky_model_save(filename, &temp, &error);
        }
    }

    return error;
}
