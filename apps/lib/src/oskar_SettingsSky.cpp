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

#include "apps/lib/oskar_SettingsSky.h"
#include <QtCore/QSettings>
#include <cstdio>
#include <ctime>
#include <cmath>

void oskar_SettingsSky::load(const QSettings& settings)
{
    // Sky model file.
    sky_file_      = settings.value("sky/source_file", "").toString();

    // Generator settings.
    generator_     = settings.value("sky/generator", "").toString();
    healpix_nside_ = settings.value("sky/generator/healpix/nside", 0).toInt();
    random_num_sources_ = (int)settings.value("sky/generator/random/num_sources", 0).toDouble();
    random_flux_density_min_ = settings.value("sky/generator/random/flux_density_min", 0.0).toDouble();
    random_flux_density_max_ = settings.value("sky/generator/random/flux_density_max", 0.0).toDouble();
    random_threshold_ = settings.value("sky/generator/random/threshold", 0.0).toDouble();
    if (generator_.toUpper() == "RANDOM_POWER_LAW")
    {
        if (settings.contains("sky/generator/random/power") && settings.contains("sky/generator/random/power1"))
            printf("== WARNING: both 'power' and 'power1' keywords detected, these can conflict!\n");
    }
    random_power1_ = settings.value("sky/generator/random/power", 0.0).toDouble();
    random_power1_ = settings.value("sky/generator/random/power1", 0.0).toDouble();
    random_power2_ = settings.value("sky/generator/random/power2", 0.0).toDouble();
    QString seed = settings.value("sky/generator/random/seed").toString();
    if (seed.toUpper() == "TIME")
        random_seed_ = (unsigned)time(NULL);
    else
        random_seed_ = (unsigned)seed.toDouble();

    output_sky_file_ = settings.value("sky/output_sky_file", "").toString();

    // Filter settings.
    filter_inner_rad_ = settings.value("sky/filter/radius_inner_deg", -1.0).toDouble() * M_PI / 180;
    filter_outer_rad_ = settings.value("sky/filter/radius_outer_deg", 1000.0).toDouble() * M_PI / 180;
    filter_flux_min_  = settings.value("sky/filter/flux_min", 0.0).toDouble();
    filter_flux_max_  = settings.value("sky/filter/flux_max", 0.0).toDouble();

    // Noise model settings.
    noise_model_ = settings.value("sky/noise_model", "").toString();
    noise_spectral_index_ = settings.value("sky/noise_model/spectral_index", 0.0).toDouble();
    seed = settings.value("sky/noise_model/seed").toString();
    if (seed.toUpper() == "TIME")
        noise_seed_ = (unsigned)time(NULL);
    else
        noise_seed_ = (unsigned)seed.toDouble();
}

void oskar_SettingsSky::print_summary() const
{
    printf("\n\n-------------------------------------------\n");
    printf("[Sky]\n");
    if (!sky_file_.isEmpty())
        printf("- sky file  = %s\n",sky_file_.toAscii().data());

    if (!generator_.isEmpty())
    {
        printf("- generator = %s\n", generator_.toAscii().data());
        if (generator_.toUpper() == "HEALPIX")
        {
            printf("  - nside = %i\n", healpix_nside_);
        }
        else if (generator_.toUpper() == "RANDOM_POWER_LAW")
        {
            printf("  - no. sources = %i\n", random_num_sources_);
            printf("  - flux density (min, max) = %e, %e\n",
                    random_flux_density_min_, random_flux_density_max_);
            printf("  - power = %f\n", random_power1_);
            printf("  - seed = %i\n", random_seed_);
        }
        else if (generator_.toUpper() == "RANDOM_BROKEN_POWER_LAW")
        {
            printf("  - no. sources = %i\n", random_num_sources_);
            printf("  - flux density (min, max) = %e, %e\n",
                    random_flux_density_min_, random_flux_density_max_);
            printf("  - threshold = %f\n", random_threshold_);
            printf("  - power1 = %f\n", random_power1_);
            printf("  - power2 = %f\n", random_power2_);
            printf("  - seed = %i\n", random_seed_);
        }
        else
            printf("   - WARNING: unrecognised generator options!\n");
    }

    if (!noise_model_.isEmpty())
    {
        printf("- noise model = %s\n", noise_model_.toAscii().data());
        printf("   - spectral index = %f\n", noise_spectral_index_);
        printf("   - seed = %i\n", noise_seed_);
    }
    printf("-------------------------------------------\n\n");
}
