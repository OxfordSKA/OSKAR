/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_settings_load_telescope.h>
#include <oskar_telescope.h>

#include <oskar_cmath.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <QtCore/QSettings>
#include <QtCore/QFileInfo>
#include <QtCore/QByteArray>
#include <QtCore/QVariant>
#include <QtCore/QString>
#include <QtCore/QStringList>

#define D2R (M_PI/180.0)

static int get_seed(const QVariant& t)
{
    return (t.toString().toLower() == "time" || t.toInt() < 1) ?
            (int)time(NULL) : t.toInt();
}

extern "C"
void oskar_settings_load_telescope(oskar_SettingsTelescope* tel,
        const char* filename, int* status)
{
    QByteArray t;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("telescope");

    // Telescope input directory.
    t = s.value("input_directory", "").toByteArray();
    if (t.size() > 0)
    {
        tel->input_directory = (char*)malloc(t.size() + 1);
        strcpy(tel->input_directory, t.constData());
    }

    // Telescope location.
    tel->longitude_rad = s.value("longitude_deg", 0.0).toDouble() * D2R;
    tel->latitude_rad  = s.value("latitude_deg", 0.0).toDouble() * D2R;
    tel->altitude_m    = s.value("altitude_m", 0.0).toDouble();

    // Station type.
    temp = s.value("station_type", "Aperture array").toString();
    if (temp.startsWith("A", Qt::CaseInsensitive))
        tel->station_type = OSKAR_STATION_TYPE_AA;
    else if (temp.startsWith("I", Qt::CaseInsensitive))
        tel->station_type = OSKAR_STATION_TYPE_ISOTROPIC;
    else if (temp.startsWith("G", Qt::CaseInsensitive))
        tel->station_type = OSKAR_STATION_TYPE_GAUSSIAN_BEAM;
    else if (temp.startsWith("VLA (PBCOR)", Qt::CaseInsensitive))
        tel->station_type = OSKAR_STATION_TYPE_VLA_PBCOR;
    else
    {
        *status = OSKAR_ERR_SETTINGS_TELESCOPE;
        return;
    }

    // Normalise beam.
    tel->normalise_beams_at_phase_centre =
            s.value("normalise_beams_at_phase_centre", true).toBool();

    // Polarisation mode.
    temp = s.value("pol_mode", "Full").toString();
    if (temp.startsWith("F", Qt::CaseInsensitive))
        tel->pol_mode = OSKAR_POL_MODE_FULL;
    else if (temp.startsWith("S", Qt::CaseInsensitive))
        tel->pol_mode = OSKAR_POL_MODE_SCALAR;

    // Duplicate first station beam if possible.
    tel->allow_station_beam_duplication =
            s.value("allow_station_beam_duplication", false).toBool();

    // Aperture array settings.
    s.beginGroup("aperture_array");
    {
        oskar_SettingsApertureArray* aa = &tel->aperture_array;
        s.beginGroup("array_pattern");
        {
            oskar_SettingsArrayPattern* ap = &aa->array_pattern;
            ap->enable = s.value("enable", true).toBool();
            ap->normalise = s.value("normalise", false).toBool();

            // Array element settings (overrides).
            s.beginGroup("element");
            {
                oskar_SettingsArrayElement* ae = &aa->array_pattern.element;

                temp = s.value("apodisation_type", "None").toString();
                if (temp.startsWith("N", Qt::CaseInsensitive))
                    ae->apodisation_type = 0;
                else
                {
                    *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                    return;
                }

                ae->gain = s.value("gain", 0.0).toDouble();
                ae->gain_error_fixed =
                        s.value("gain_error_fixed", 0.0).toDouble();
                ae->gain_error_time =
                        s.value("gain_error_time", 0.0).toDouble();
                ae->phase_error_fixed_rad =
                        s.value("phase_error_fixed_deg", 0.0).toDouble() * D2R;
                ae->phase_error_time_rad =
                        s.value("phase_error_time_deg", 0.0).toDouble() * D2R;
                ae->position_error_xy_m =
                        s.value("position_error_xy_m", 0.0).toDouble();
                ae->x_orientation_error_rad =
                        s.value("x_orientation_error_deg", 0.0).toDouble() * D2R;
                ae->y_orientation_error_rad =
                        s.value("y_orientation_error_deg", 0.0).toDouble() * D2R;

                // Station element random seeds.
                ae->seed_gain_errors = get_seed(s.value("seed_gain_errors", 1));
                ae->seed_phase_errors = get_seed(s.value("seed_phase_errors", 1));
                ae->seed_time_variable_errors =
                        get_seed(s.value("seed_time_variable_errors", 1));
                ae->seed_position_xy_errors =
                        get_seed(s.value("seed_position_xy_errors", 1));
                ae->seed_x_orientation_error =
                        get_seed(s.value("seed_x_orientation_error", 1));
                ae->seed_y_orientation_error =
                        get_seed(s.value("seed_y_orientation_error", 1));
            }
            s.endGroup(); // End array element group.
        }
        s.endGroup(); // End array pattern group.

        // Element pattern settings.
        s.beginGroup("element_pattern");
        {
            oskar_SettingsElementPattern* ep = &aa->element_pattern;
            ep->enable_numerical_patterns =
                    s.value("enable_numerical", true).toBool();

            temp = s.value("functional_type", "Dipole").toString();
            if (temp.startsWith("D", Qt::CaseInsensitive))
                ep->functional_type = OSKAR_ELEMENT_TYPE_DIPOLE;
            else if (temp.startsWith("G", Qt::CaseInsensitive))
                ep->functional_type = OSKAR_ELEMENT_TYPE_GEOMETRIC_DIPOLE;
            else if (temp.startsWith("I", Qt::CaseInsensitive))
                ep->functional_type = OSKAR_ELEMENT_TYPE_ISOTROPIC;
            else
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }

            ep->dipole_length = s.value("dipole_length", 0.5).toDouble();
            temp = s.value("dipole_length_units", "Wavelengths").toString();
            if (temp.startsWith("W", Qt::CaseInsensitive))
                ep->dipole_length_units = OSKAR_WAVELENGTHS;
            else if (temp.startsWith("M", Qt::CaseInsensitive))
                ep->dipole_length_units = OSKAR_METRES;
            else
            {
                *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                return;
            }

            s.beginGroup("taper");
            {
                temp = s.value("type", "None").toString();
                if (temp.startsWith("N", Qt::CaseInsensitive))
                    ep->taper.type = OSKAR_ELEMENT_TAPER_NONE;
                else if (temp.startsWith("C", Qt::CaseInsensitive))
                    ep->taper.type = OSKAR_ELEMENT_TAPER_COSINE;
                else if (temp.startsWith("G", Qt::CaseInsensitive))
                    ep->taper.type = OSKAR_ELEMENT_TAPER_GAUSSIAN;
                else
                {
                    *status = OSKAR_ERR_SETTINGS_TELESCOPE;
                    return;
                }
                ep->taper.cosine_power =
                        s.value("cosine_power", 1.0).toDouble();
                ep->taper.gaussian_fwhm_rad =
                        s.value("gaussian_fwhm_deg", 45.0).toDouble() * D2R;
            }
            s.endGroup(); // End taper group.
        }
        s.endGroup(); // End element pattern group.
    }
    s.endGroup(); // End aperture array group.

    // Gaussian beam settings.
    s.beginGroup("gaussian_beam");
    {
        // NOTE: this sort of error check should be automated and
        // probably should use the log to ensure common formatting.
        // **DONT FIX** until some design thought has been made to how to
        // deal with settings functions and stdout messages / error codes.
        QStringList keys = s.allKeys();
        if (tel->station_type == OSKAR_STATION_TYPE_GAUSSIAN_BEAM &&
            (!keys.contains("fwhm_deg") || !keys.contains("ref_freq_hz")))
        {
            printf("E|\n");
            printf("E|== ERROR: One or more required Gaussian Beam settings "
                    "not set.\n");
            printf("E|\n");
            *status = OSKAR_ERR_SETTINGS_TELESCOPE;
            return;
        }
        tel->gaussian_beam.fwhm_deg = s.value("fwhm_deg", 1.0).toDouble();
        tel->gaussian_beam.ref_freq_hz = s.value("ref_freq_hz", 0.0).toDouble();
    }
    s.endGroup(); // End Gaussian beam group

    // Telescope output directory.
    t = s.value("output_directory", "").toByteArray();
    if (t.size() > 0)
    {
        tel->output_directory = (char*)malloc(t.size() + 1);
        strcpy(tel->output_directory, t.constData());
    }
}
