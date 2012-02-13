/*
 * Copyright (c) 2011, The University of Oxford
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
#include "math/oskar_healpix_nside_to_npix.h"
#include "math/oskar_healpix_pix_to_angles_ring.h"
#include "math/oskar_random_power_law.h"
#include "math/oskar_random_broken_power_law.h"
#include "sky/oskar_generate_random_coordinate.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

extern "C"
oskar_SkyModel* oskar_set_up_sky(const oskar_Settings* settings)
{
    const char* filename = NULL;
    int type, err;

    // Create empty sky model.
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_SkyModel *sky = new oskar_SkyModel(type, OSKAR_LOCATION_CPU);

    // Load sky file if it exists.
    filename = settings->sky.input_sky_file;
    if (filename)
    {
        if (strlen(filename) > 0)
        {
            // Load the sky model data into a temporary sky model.
            oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
            printf("--> Loading OSKAR sky model data... ");
            fflush(stdout);
            err = temp.load(filename);
            if (err)
            {
                delete sky;
                return NULL;
            }

            // Get the filter parameters.
            double inner = settings->sky.input_sky_filter.radius_inner;
            double outer = settings->sky.input_sky_filter.radius_outer;
            double flux_min = settings->sky.input_sky_filter.flux_min;
            double flux_max = settings->sky.input_sky_filter.flux_max;

            // Apply filters.
            temp.filter_by_flux(flux_min, flux_max);
            temp.filter_by_radius(inner, outer,	settings->obs.ra0_rad,
                    settings->obs.dec0_rad);
            printf("done.\n");

            // Save the new model sky.
            sky->append(&temp);
            if (err)
            {
                delete sky;
                return NULL;
            }
        }
    }

    // Load GSM file if it exists.
    filename = settings->sky.gsm_file;
    if (filename)
    {
        if (strlen(filename) > 0)
        {
            // Load the sky model data into a temporary sky model.
            oskar_SkyModel temp(type, OSKAR_LOCATION_CPU);
            printf("--> Loading GSM data... ");
            fflush(stdout);
            err = temp.load_gsm(filename);
            if (err)
            {
                delete sky;
                return NULL;
            }

            // Get the filter parameters.
            double inner = settings->sky.gsm_filter.radius_inner;
            double outer = settings->sky.gsm_filter.radius_outer;
            double flux_min = settings->sky.gsm_filter.flux_min;
            double flux_max = settings->sky.gsm_filter.flux_max;

            // Apply filters.
            temp.filter_by_flux(flux_min, flux_max);
            temp.filter_by_radius(inner, outer,	settings->obs.ra0_rad,
                    settings->obs.dec0_rad);
            printf("done.\n");

            // Save the new model sky.
            sky->append(&temp);
            if (err)
            {
                delete sky;
                return NULL;
            }
        }
    }

    // HEALPix generator.
    if (settings->sky.generator.healpix.nside != 0)
    {
        // Get the generator parameters.
        int nside = settings->sky.generator.healpix.nside;
        int npix = oskar_healpix_nside_to_npix(nside);

        // Generate the new positions into a temporary sky model.
        oskar_SkyModel temp(type, OSKAR_LOCATION_CPU, npix);
        printf("--> Generating HEALPIX source positions... ");
        fflush(stdout);
        #pragma omp parallel for
        for (int i = 0; i < npix; ++i)
        {
            double ra, dec;
            oskar_healpix_pix_to_angles_ring(nside, i, &dec, &ra);
            dec = M_PI / 2.0 - dec;
            temp.set_source(i, ra, dec, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Get the filter parameters.
        double inner = settings->sky.generator.healpix.filter.radius_inner;
        double outer = settings->sky.generator.healpix.filter.radius_outer;
        double flux_min = settings->sky.generator.healpix.filter.flux_min;
        double flux_max = settings->sky.generator.healpix.filter.flux_max;

        // Apply filters.
        temp.filter_by_flux(flux_min, flux_max);
        temp.filter_by_radius(inner, outer,	settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        printf("done.\n");

        // Save the new model sky.
        sky->append(&temp);
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
        printf("--> Generating random power law source distribution... ");
        fflush(stdout);
        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_power_law(min, max, power);
            temp.set_source(i, ra, dec, b, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Get the filter parameters.
        double inner = settings->sky.generator.random_power_law.filter.radius_inner;
        double outer = settings->sky.generator.random_power_law.filter.radius_outer;
        double flux_min = settings->sky.generator.random_power_law.filter.flux_min;
        double flux_max = settings->sky.generator.random_power_law.filter.flux_max;

        // Apply filters.
        temp.filter_by_flux(flux_min, flux_max);
        temp.filter_by_radius(inner, outer,	settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        printf("done.\n");

        // Save the new model sky.
        sky->append(&temp);
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
        printf("--> Generating random broken power law source distribution... ");
        fflush(stdout);
        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_broken_power_law(min, max, threshold, power1, power2);
            temp.set_source(i, ra, dec, b, 0.0, 0.0, 0.0, 0.0, 0.0);
        }

        // Get the filter parameters.
        double inner = settings->sky.generator.random_broken_power_law.filter.radius_inner;
        double outer = settings->sky.generator.random_broken_power_law.filter.radius_outer;
        double flux_min = settings->sky.generator.random_broken_power_law.filter.flux_min;
        double flux_max = settings->sky.generator.random_broken_power_law.filter.flux_max;

        // Apply filters.
        temp.filter_by_flux(flux_min, flux_max);
        temp.filter_by_radius(inner, outer,	settings->obs.ra0_rad,
                settings->obs.dec0_rad);
        printf("done.\n");

        // Save the new model sky.
        sky->append(&temp);
    }

    // Compute source direction cosines relative to phase centre.
    printf("--> Computing source direction cosines... ");
    fflush(stdout);
    err = sky->compute_relative_lmn(settings->obs.ra0_rad,
            settings->obs.dec0_rad);
    if (err)
    {
        delete sky;
        return NULL;
    }
    printf("done.\n");

    // Write sky model out.
    filename = settings->sky.output_sky_file;
    if (filename)
    {
        if (strlen(filename))
        {
            printf("--> Writing sky file to disk as: %s\n", filename);
            sky->write(filename);
        }
    }

    // Check if sky model contains no sources.
    if (sky->num_sources == 0)
        fprintf(stderr, "--> WARNING: Sky model contains no sources.\n");
    else
    {
        // Print summary data.
        printf("\n");
        printf("= Sky model\n");
        printf("  - Num. sources           = %u\n", sky->num_sources);
        printf("\n");
    }

    // Return the structure.
    return sky;
}
