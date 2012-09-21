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

#include "apps/lib/oskar_settings_load_sky.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QStringList>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define D2R M_PI/180.0
#define ARCSEC2RAD M_PI/648000.0

extern "C"
int oskar_settings_load_sky(oskar_SettingsSky* sky, const char* filename)
{
    QByteArray t;
    QStringList list;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("sky");

    // Output OSKAR sky model text file.
    t = s.value("output_text_file").toByteArray();
    if (t.size() > 0)
    {
        sky->output_text_file = (char*)malloc(t.size() + 1);
        strcpy(sky->output_text_file, t.constData());
    }

    // Output OSKAR sky model binary file.
    t = s.value("output_binary_file").toByteArray();
    if (t.size() > 0)
    {
        sky->output_binary_file = (char*)malloc(t.size() + 1);
        strcpy(sky->output_binary_file, t.constData());
    }

    // Input OSKAR sky model files.
    list = s.value("oskar_source_file").toStringList();
    sky->num_sky_files = list.size();
    sky->input_sky_file = (char**)malloc(sky->num_sky_files * sizeof(char*));
    for (int i = 0; i < sky->num_sky_files; ++i)
    {
        t = list[i].toAscii();
        sky->input_sky_file[i] = (char*)malloc(t.size() + 1);
        strcpy(sky->input_sky_file[i], t.constData());
    }

    // Input OSKAR sky model filter.
    s.beginGroup("oskar_source_file");
    s.beginGroup("filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->input_sky_filter.flux_min = 0.0;
    else
        sky->input_sky_filter.flux_min = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->input_sky_filter.flux_max = 0.0;
    else
        sky->input_sky_filter.flux_max = temp.toDouble();
    sky->input_sky_filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->input_sky_filter.radius_outer =
            s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
    s.beginGroup("extended_sources");
    sky->input_sky_extended_sources.FWHM_major =
            s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    sky->input_sky_extended_sources.FWHM_minor =
                s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    sky->input_sky_extended_sources.position_angle =
                s.value("position_angle").toDouble() * D2R;
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
    temp = s.value("flux_min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->gsm_filter.flux_min = 0.0;
    else
        sky->gsm_filter.flux_min = temp.toDouble();
    temp = s.value("flux_max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->gsm_filter.flux_max = 0.0;
    else
        sky->gsm_filter.flux_max = temp.toDouble();
    sky->gsm_filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->gsm_filter.radius_outer =
            s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
    s.beginGroup("extended_sources");
    sky->gsm_extended_sources.FWHM_major =
            s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    sky->gsm_extended_sources.FWHM_minor =
                s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    sky->gsm_extended_sources.position_angle =
                s.value("position_angle").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // Input FITS files.
    list = s.value("fits_file").toStringList();
    sky->num_fits_files = list.size();
    sky->fits_file = (char**)malloc(sky->num_fits_files * sizeof(char*));
    for (int i = 0; i < sky->num_fits_files; ++i)
    {
        t = list[i].toAscii();
        sky->fits_file[i] = (char*)malloc(t.size() + 1);
        strcpy(sky->fits_file[i], t.constData());
    }

    // FITS import settings.
    s.beginGroup("fits_file");
    sky->fits_file_settings.downsample_factor =
            s.value("downsample_factor", 1).toInt();
    sky->fits_file_settings.min_peak_fraction =
            s.value("min_peak_fraction", 0.02).toDouble();
    sky->fits_file_settings.noise_floor =
            s.value("noise_floor", 0.0).toDouble();
    sky->fits_file_settings.spectral_index =
            s.value("spectral_index", 0.0).toDouble();
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
    sky->generator.random_power_law.seed = (temp.toUpper() == "TIME" ||
            temp.toInt() < 0) ? (int)time(NULL) : temp.toInt();
    // Random power-law generator filter.
    s.beginGroup("filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->generator.random_power_law.filter.flux_min = 0.0;
    else
        sky->generator.random_power_law.filter.flux_min = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->generator.random_power_law.filter.flux_max = 0.0;
    else
        sky->generator.random_power_law.filter.flux_max = temp.toDouble();
    sky->generator.random_power_law.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.random_power_law.filter.radius_outer =
            s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
    s.beginGroup("extended_sources");
    sky->generator.random_power_law.extended_sources.FWHM_major =
            s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    sky->generator.random_power_law.extended_sources.FWHM_minor =
                s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    sky->generator.random_power_law.extended_sources.position_angle =
                s.value("position_angle").toDouble() * D2R;
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
    sky->generator.random_broken_power_law.seed = (temp.toUpper() == "TIME" ||
            temp.toInt() < 0) ? (int)time(NULL) : temp.toInt();
    // Random broken-power-law generator filter.
    s.beginGroup("filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->generator.random_broken_power_law.filter.flux_min = 0.0;
    else
        sky->generator.random_power_law.filter.flux_min = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->generator.random_broken_power_law.filter.flux_max = 0.0;
    else
        sky->generator.random_broken_power_law.filter.flux_max = temp.toDouble();
    sky->generator.random_broken_power_law.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.random_broken_power_law.filter.radius_outer =
            s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
    s.beginGroup("extended_sources");
    sky->generator.random_broken_power_law.extended_sources.FWHM_major =
            s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    sky->generator.random_broken_power_law.extended_sources.FWHM_minor =
                s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    sky->generator.random_broken_power_law.extended_sources.position_angle =
                s.value("position_angle").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // HEALPix generator settings.
    s.beginGroup("healpix");
    sky->generator.healpix.nside = s.value("nside", 0).toInt();
    // HEALPix generator filter.
    s.beginGroup("filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->generator.healpix.filter.flux_min = 0.0;
    else
        sky->generator.healpix.filter.flux_min = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->generator.healpix.filter.flux_max = 0.0;
    else
        sky->generator.healpix.filter.flux_max = temp.toDouble();
    sky->generator.healpix.filter.radius_inner =
            s.value("radius_inner_deg").toDouble() * D2R;
    sky->generator.healpix.filter.radius_outer =
            s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
    s.beginGroup("extended_sources");
    sky->generator.healpix.extended_sources.FWHM_major =
            s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    sky->generator.healpix.extended_sources.FWHM_minor =
                s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    sky->generator.healpix.extended_sources.position_angle =
                s.value("position_angle").toDouble() * D2R;
    s.endGroup();
    s.endGroup();

    // End generator group.
    s.endGroup();

    return OSKAR_SUCCESS;
}
