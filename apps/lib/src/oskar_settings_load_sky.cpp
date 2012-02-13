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

#include "apps/lib/oskar_settings_load_sky.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>

#define D2R M_PI/180.0

extern "C"
int oskar_settings_load_sky(oskar_SettingsSkyNew* sky, const char* filename)
{
    QByteArray t;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("sky");

    // Output OSKAR sky model file.
    t = s.value("output_sky_file").toByteArray();
    if (t.size() > 0)
    {
        sky->output_sky_file = (char*)malloc(t.size() + 1);
        strcpy(sky->output_sky_file, t.constData());
    }

    // Input OSKAR sky model file.
    t = s.value("oskar_source_file").toByteArray();
    if (t.size() > 0)
    {
        sky->input_sky_file = (char*)malloc(t.size() + 1);
        strcpy(sky->input_sky_file, t.constData());
    }

    // Input OSKAR sky model filter.
    s.beginGroup("oskar_source_file");
    s.beginGroup("filter");
    sky->input_sky_filter.flux_min = s.value("flux_min").toDouble();
    sky->input_sky_filter.flux_max = s.value("flux_max").toDouble();
    sky->input_sky_filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->input_sky_filter.radius_outer =
            s.value("radius_outer_deg").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // GSM file.
    t = s.value("gsm_file").toByteArray();
    if (t.size() > 0)
    {
        sky->gsm_file = (char*)malloc(t.size() + 1);
        strcpy(sky->gsm_file, t.constData());
    }

    // GSM filter.
    s.beginGroup("gsm_file");
    s.beginGroup("filter");
    sky->gsm_filter.flux_min = s.value("flux_min").toDouble();
    sky->gsm_filter.flux_max = s.value("flux_max").toDouble();
    sky->gsm_filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->gsm_filter.radius_outer =
            s.value("radius_outer_deg").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // Generator settings.
    s.beginGroup("generator");

    // Random power-law generator settings.
    s.beginGroup("random_power_law");
    sky->generator.random_power_law.num_sources =
            s.value("num_sources").toInt();
    sky->generator.random_power_law.flux_min = s.value("flux_min").toDouble();
    sky->generator.random_power_law.flux_max = s.value("flux_max").toDouble();
    sky->generator.random_power_law.power = s.value("power").toDouble();
    temp = s.value("seed").toString();
    sky->generator.random_power_law.seed = (temp.toUpper() == "TIME")
            ? (int)time(NULL) : (int)temp.toDouble();
    // Random power-law generator filter.
    s.beginGroup("filter");
    sky->generator.random_power_law.filter.flux_min =
            s.value("flux_min").toDouble();
    sky->generator.random_power_law.filter.flux_max =
            s.value("flux_max").toDouble();
    sky->generator.random_power_law.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.random_power_law.filter.radius_outer =
            s.value("radius_outer_deg").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // Random broken-power-law generator settings.
    s.beginGroup("random_broken_power_law");
    sky->generator.random_broken_power_law.num_sources =
            s.value("num_sources").toInt();
    sky->generator.random_broken_power_law.flux_min =
            s.value("flux_min").toDouble();
    sky->generator.random_broken_power_law.flux_max =
            s.value("flux_max").toDouble();
    sky->generator.random_broken_power_law.threshold =
            s.value("threshold").toDouble();
    sky->generator.random_broken_power_law.power1 =
            s.value("power1").toDouble();
    sky->generator.random_broken_power_law.power2 =
            s.value("power2").toDouble();
    temp = s.value("seed").toString();
    sky->generator.random_broken_power_law.seed = (temp.toUpper() == "TIME")
            ? (int)time(NULL) : (int)temp.toDouble();
    // Random broken-power-law generator filter.
    s.beginGroup("filter");
    sky->generator.random_broken_power_law.filter.flux_min =
            s.value("flux_min").toDouble();
    sky->generator.random_broken_power_law.filter.flux_max =
            s.value("flux_max").toDouble();
    sky->generator.random_broken_power_law.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.random_broken_power_law.filter.radius_outer =
            s.value("radius_outer_deg").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // HEALPix generator settings.
    s.beginGroup("healpix");
    sky->generator.healpix.nside = s.value("nside", 0).toInt();
    // HEALPix generator filter.
    s.beginGroup("filter");
    sky->generator.healpix.filter.flux_min = s.value("flux_min").toDouble();
    sky->generator.healpix.filter.flux_max = s.value("flux_max").toDouble();
    sky->generator.healpix.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.healpix.filter.radius_outer =
            s.value("radius_outer_deg").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // End generator group.
    s.endGroup();

    // Noise model settings.
    sky->noise_model.type = -1;
    temp = s.value("noise_model").toString();
    if (temp.toUpper() == "VLA_MEMO_146")
        sky->noise_model.type = OSKAR_NOISE_VLA_MEMO_146;
    sky->noise_model.spectral_index =
            s.value("noise_model/spectral_index").toDouble();
    temp = s.value("noise_model/seed").toString();
    sky->noise_model.seed = (temp.toUpper() == "TIME")
            ? (int)time(NULL) : (int)temp.toDouble();

    return 0;
}
