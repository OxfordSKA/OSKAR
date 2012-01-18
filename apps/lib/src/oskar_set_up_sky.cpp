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
#include "apps/lib/oskar_SettingsSky.h"
#include "math/oskar_healpix_nside_to_npix.h"
#include "math/oskar_healpix_pix_to_angles_ring.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "math/oskar_random_power_law.h"
#include "math/oskar_random_broken_power_law.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <QtCore/QByteArray>

extern "C"
oskar_SkyModel* oskar_set_up_sky(const oskar_Settings& settings)
{
    int type, err;

    // Create empty sky model.
    type = settings.double_precision() ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_SkyModel *sky = new oskar_SkyModel(type, OSKAR_LOCATION_CPU);

    // Load sky file if it exists.
    QByteArray sky_file = settings.sky().sky_file().toAscii();
    if (!sky_file.isEmpty())
    {
        printf("--> Loading sky model data... ");
        fflush(stdout);
        err = sky->load(sky_file);
        printf("done.\n");
        if (err)
        {
            delete sky;
            return NULL;
        }
    }

    // TODO: enable 2 generates at the same time somehow?

    // Set up sky using generator parameters.
    if (settings.sky().generator().toUpper() == "HEALPIX")
    {
        int nside = settings.sky().healpix_nside();
        int npix = oskar_healpix_nside_to_npix(nside);

        // Add enough positions to the sky model.
        int old_size = sky->num_sources;
        sky->resize(old_size + npix);

        // Generate the new positions.
        printf("--> Generating HEALPIX source positions... ");
        fflush(stdout);
        #pragma omp parallel for
        for (int i = 0; i < npix; ++i)
        {
            double ra, dec;
            oskar_healpix_pix_to_angles_ring(nside, i, &dec, &ra);
            dec = M_PI / 2.0 - dec;
            sky->set_source(i + old_size, ra, dec,
                    1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        printf("done.\n");
    }
    else if (settings.sky().generator().toUpper() == "RANDOM_POWER_LAW")
    {
        printf("--> Generating random power law source distribution... ");

        int old_size = sky->num_sources;
        int num_sources = settings.sky().random_num_sources();
        sky->resize(old_size + num_sources);

        double min = settings.sky().random_flux_density_min();
        double max = settings.sky().random_flux_density_max();
        double power = settings.sky().random_power();

        srand(settings.sky().random_seed());

        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_power_law(min, max, power);
            sky->set_source(i + old_size, ra, dec, b, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        printf("done.\n");
    }
    else if (settings.sky().generator().toUpper() == "RANDOM_BROKEN_POWER_LAW")
    {
        printf("--> Generating random power broken law source distribution...");
        int old_size = sky->num_sources;
        int num_sources = settings.sky().random_num_sources();
        sky->resize(old_size + num_sources);

        double min = settings.sky().random_flux_density_min();
        double max = settings.sky().random_flux_density_max();
        double power1 = settings.sky().random_power1();
        double power2 = settings.sky().random_power2();
        double threshold = settings.sky().random_threshold();

        srand(settings.sky().random_seed());

        for (int i = 0; i < num_sources; ++i)
        {
            double ra, dec, b;
            oskar_generate_random_coordinate(&ra, &dec);
            b = oskar_random_broken_power_law(min, max, threshold, power1, power2);
            sky->set_source(i + old_size, ra, dec, b, 0.0, 0.0, 0.0, 0.0, 0.0);
        }
        printf("done.\n");
    }

    // Compute source direction cosines relative to phase centre.
    printf("--> Computing source direction cosines... ");
    fflush(stdout);
    err = sky->compute_relative_lmn(settings.obs().ra0_rad(),
            settings.obs().dec0_rad());
    printf("done.\n");
    if (err)
    {
        delete sky;
        return NULL;
    }

    // Print summary data.
    printf("\n");
    printf("= Sky model\n");
    printf("  - Sky file               = %s\n", sky_file.constData());
    printf("  - Num. sources           = %u\n", sky->num_sources);
    printf("\n");

    // Check if sky model contains no sources.
    if (sky->num_sources == 0)
    {
        fprintf(stderr, "--> WARNING: Sky model contains no sources.\n");
    }

    if (!settings.sky().output_sky_file().isEmpty())
    {
        printf("--> Writing sky file to disk as: %s\n",
                settings.sky().output_sky_file().toAscii().constData());
        sky->write(settings.sky().output_sky_file().toAscii().constData());
    }

    // Return the structure.
    return sky;
}
