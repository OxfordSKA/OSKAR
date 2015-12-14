/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_settings_load_sky.h>

#include <oskar_cmath.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cfloat>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QStringList>

#define D2R M_PI/180.0
#define ARCSEC2RAD M_PI/648000.0

static void get_pol_params(QSettings& s, oskar_SettingsSkyPolarisation* ext);
static void get_extended_params(QSettings& s, oskar_SettingsSkyExtendedSources* ext);
static void get_filter_params(QSettings& s, oskar_SettingsSkyFilter* flt);
static int get_seed(const QVariant& t);

extern "C"
void oskar_settings_load_sky(oskar_SettingsSky* sky, const char* filename,
        int* status)
{
    QByteArray t;
    QStringList list;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);

    // Check if safe to proceed.
    if (*status) return;

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
    s.beginGroup("oskar_sky_model");
    QVariant vFiles = s.value("file");
    if (vFiles.type() == QVariant::StringList)
        list = vFiles.toStringList();
    else if (vFiles.type() == QVariant::String)
        list = vFiles.toString().split(",");
    //list = s.value("file").toStringList();
    sky->oskar_sky_model.num_files = list.size();
    sky->oskar_sky_model.file = (char**)malloc(sky->oskar_sky_model.num_files *
            sizeof(char*));
    for (int i = 0; i < sky->oskar_sky_model.num_files; ++i)
    {
        t = list[i].toLatin1();
        sky->oskar_sky_model.file[i] = (char*)malloc(t.size() + 1);
        strcpy(sky->oskar_sky_model.file[i], t.constData());
    }
    get_filter_params(s, &sky->oskar_sky_model.filter);
    get_extended_params(s, &sky->oskar_sky_model.extended_sources);
    s.endGroup();

    // FIXME GSM file - needs reference frequency.
//    s.beginGroup("gsm");
//    t = s.value("file").toByteArray();
//    if (t.size() > 0)
//    {
//        sky->gsm.file = (char*)malloc(t.size() + 1);
//        strcpy(sky->gsm.file, t.constData());
//    }
//    get_filter_params(s, &sky->gsm.filter);
//    get_extended_params(s, &sky->gsm.extended_sources);
//    s.endGroup();

    // Input FITS image files.
    s.beginGroup("fits_image");
    list = s.value("file").toStringList();
    sky->fits_image.num_files = list.size();
    sky->fits_image.file = (char**)malloc(sky->fits_image.num_files *
            sizeof(char*));
    for (int i = 0; i < sky->fits_image.num_files; ++i)
    {
        t = list[i].toLatin1();
        sky->fits_image.file[i] = (char*)malloc(t.size() + 1);
        strcpy(sky->fits_image.file[i], t.constData());
    }
    sky->fits_image.downsample_factor =
            s.value("downsample_factor", 1).toInt();
    sky->fits_image.min_peak_fraction =
            s.value("min_peak_fraction", 0.02).toDouble();
    sky->fits_image.noise_floor =
            s.value("noise_floor", 0.0).toDouble();
    sky->fits_image.spectral_index =
            s.value("spectral_index", 0.0).toDouble();
    s.endGroup();

    // Input HEALPix FITS files.
    s.beginGroup("healpix_fits");
    list = s.value("file").toStringList();
    sky->healpix_fits.num_files = list.size();
    sky->healpix_fits.file = (char**)malloc(sky->healpix_fits.num_files *
            sizeof(char*));
    for (int i = 0; i < sky->healpix_fits.num_files; ++i)
    {
        t = list[i].toLatin1();
        sky->healpix_fits.file[i] = (char*)malloc(t.size() + 1);
        strcpy(sky->healpix_fits.file[i], t.constData());
    }
    temp = s.value("coord_sys", "Galactic").toString();
    if (temp.startsWith('G', Qt::CaseInsensitive))
        sky->healpix_fits.coord_sys = OSKAR_SPHERICAL_TYPE_GALACTIC;
    else
        sky->healpix_fits.coord_sys = OSKAR_SPHERICAL_TYPE_EQUATORIAL;
    temp = s.value("map_units", "mK/sr").toString();
    if (temp.startsWith("mK", Qt::CaseInsensitive))
        sky->healpix_fits.map_units = OSKAR_MAP_UNITS_MK_PER_SR;
    else if (temp.startsWith("K", Qt::CaseInsensitive))
        sky->healpix_fits.map_units = OSKAR_MAP_UNITS_K_PER_SR;
    else
        sky->healpix_fits.map_units = OSKAR_MAP_UNITS_JY;
    get_filter_params(s, &sky->healpix_fits.filter);
    get_extended_params(s, &sky->healpix_fits.extended_sources);
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
    sky->generator.random_power_law.seed = get_seed(s.value("seed", 1));
    // Random power-law generator filter.
    get_filter_params(s, &sky->generator.random_power_law.filter);
    get_extended_params(s, &sky->generator.random_power_law.extended_sources);
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
    sky->generator.random_broken_power_law.seed = get_seed(s.value("seed", 1));
    // Random broken-power-law generator filter.
    get_filter_params(s, &sky->generator.random_broken_power_law.filter);
    get_extended_params(s, &sky->generator.random_broken_power_law.extended_sources);
    s.endGroup();

    // Grid generator settings.
    s.beginGroup("grid");
    sky->generator.grid.side_length = s.value("side_length", 0).toInt();
    sky->generator.grid.fov_rad = s.value("fov_deg", 0.0).toDouble() * D2R;
    sky->generator.grid.mean_flux_jy = s.value("mean_flux_jy", 0.0).toDouble();
    sky->generator.grid.std_flux_jy = s.value("std_flux_jy", 0.0).toDouble();
    sky->generator.grid.seed = get_seed(s.value("seed", 1));
    // Grid generator extended parameters.
    get_extended_params(s, &sky->generator.grid.extended_sources);
    get_pol_params(s, &sky->generator.grid.pol);
    s.endGroup();

    // HEALPix generator settings.
    s.beginGroup("healpix");
    sky->generator.healpix.nside = s.value("nside", 0).toInt();
    sky->generator.healpix.amplitude = s.value("amplitude", 1.0).toDouble();
    // HEALPix generator filter.
    get_filter_params(s, &sky->generator.healpix.filter);
    get_extended_params(s, &sky->generator.healpix.extended_sources);
    s.endGroup();

    // End generator group.
    s.endGroup();

    // Spectral index override settings.
    s.beginGroup("spectral_index");
    sky->spectral_index.override = s.value("override", false).toBool();
    sky->spectral_index.ref_frequency_hz =
            s.value("ref_frequency_hz", 0.0).toDouble();
    sky->spectral_index.mean = s.value("mean", 0.0).toDouble();
    sky->spectral_index.std_dev = s.value("std_dev", 0.0).toDouble();
    sky->spectral_index.seed = get_seed(s.value("seed", 1));
    s.endGroup();

    // Common flux filtering settings, applied per-channel
    // after spectral index scaling.
    s.beginGroup("common_flux_filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        sky->common_flux_filter_min_jy = 0.0;
    else
        sky->common_flux_filter_min_jy = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        sky->common_flux_filter_max_jy = FLT_MAX;
    else
        sky->common_flux_filter_max_jy = temp.toDouble();
    s.endGroup();

    sky->zero_failed_gaussians = s.value("advanced/zero_failed_gaussians", false).toBool();
    sky->apply_horizon_clip = s.value("advanced/apply_horizon_clip", true).toBool();
}

static void get_pol_params(QSettings& s, oskar_SettingsSkyPolarisation* pol)
{
    s.beginGroup("pol");
    pol->mean_pol_fraction = s.value("mean_pol_fraction").toDouble();
    pol->std_pol_fraction = s.value("std_pol_fraction").toDouble();
    pol->mean_pol_angle_rad = s.value("mean_pol_angle_deg").toDouble() * D2R;
    pol->std_pol_angle_rad = s.value("std_pol_angle_deg").toDouble() * D2R;
    pol->seed = get_seed(s.value("seed", 1));
    s.endGroup();
}

static void get_extended_params(QSettings& s,
        oskar_SettingsSkyExtendedSources* ext)
{
    s.beginGroup("extended_sources");
    ext->FWHM_major_rad = s.value("FWHM_major").toDouble() * ARCSEC2RAD;
    ext->FWHM_minor_rad = s.value("FWHM_minor").toDouble() * ARCSEC2RAD;
    ext->position_angle_rad = s.value("position_angle").toDouble() * D2R;
    s.endGroup();
}

static void get_filter_params(QSettings& s, oskar_SettingsSkyFilter* flt)
{
    QString temp;
    s.beginGroup("filter");
    temp = s.value("flux_min", "min").toString();
    if (temp.compare("min", Qt::CaseInsensitive) == 0)
        flt->flux_min = 0.0;
    else
        flt->flux_min = temp.toDouble();
    temp = s.value("flux_max", "max").toString();
    if (temp.compare("max", Qt::CaseInsensitive) == 0)
        flt->flux_max = 0.0;
    else
        flt->flux_max = temp.toDouble();
    flt->radius_inner_rad = s.value("radius_inner_deg").toDouble() * D2R;
    flt->radius_outer_rad = s.value("radius_outer_deg", 180.0).toDouble() * D2R;
    s.endGroup();
}

static int get_seed(const QVariant& t)
{
    QString str = t.toString().toUpper();
    int val = str.toInt();
    return (str == "TIME" || val < 1) ? (int)time(NULL) : val;
}
