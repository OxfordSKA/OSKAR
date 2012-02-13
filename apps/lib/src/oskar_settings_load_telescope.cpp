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

#include "apps/lib/oskar_settings_load_telescope.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QFileInfo>
#include <QtCore/QByteArray>
#include <QtCore/QVariant>
#include <QtCore/QString>

extern "C"
int oskar_settings_load_telescope(oskar_SettingsTelescope* tel,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("telescope");

    // Telescope layout file.
    t = s.value("layout_file", "").toByteArray();
    if (t.size() > 0)
    {
        tel->layout_file = (char*)malloc(t.size() + 1);
        strcpy(tel->layout_file, t.constData());

        // Check that telescope layout file exists.
        if (!QFileInfo(tel->layout_file).isFile())
        {
            fprintf(stderr, "ERROR: Telescope layout file doesn't exist!\n");
            return OSKAR_ERR_SETTINGS;
        }
    }
    else
    {
        fprintf(stderr, "ERROR: No telescope layout file specified!\n");
        return OSKAR_ERR_SETTINGS;
    }

    // Station directory.
    t = s.value("station_directory", "").toByteArray();
    if (t.size() > 0)
    {
        tel->station_dir = (char*)malloc(t.size() + 1);
        strcpy(tel->station_dir, t.constData());

        // Check that station directory exists.
        if (!QFileInfo(tel->station_dir).isDir())
        {
            fprintf(stderr, "ERROR: Station directory doesn't exist!\n");
            return OSKAR_ERR_SETTINGS;
        }
    }
    else
    {
        fprintf(stderr, "ERROR: No station directory specified!\n");
        return OSKAR_ERR_SETTINGS;
    }

    // Telescope location.
    tel->longitude_rad = s.value("longitude_deg", 0.0).toDouble() * M_PI / 180.0;
    tel->latitude_rad  = s.value("latitude_deg", 0.0).toDouble() * M_PI / 180.0;
    tel->altitude_m    = s.value("altitude_m", 0.0).toDouble();

    // Station settings.
    tel->enable_station_beam = s.value("enable_station_beam", true).toBool();
    tel->normalise_station_beam = s.value("normalise_station_beam", false).toBool();

    // Station element settings (overrides).
    tel->element_amp_gain = s.value("element_amp_gain", 1.0).toDouble();
    tel->element_amp_error = s.value("element_amp_error", 0.0).toDouble();
    tel->element_phase_offset_rad = s.value("element_phase_offset_deg", 0.0).toDouble() * M_PI / 180.0;
    tel->element_phase_error_rad = s.value("element_phase_error_deg", 0.0).toDouble() * M_PI / 180.0;

    // Receiver temperature.
    tel->receiver_temperature = s.value("receiver_temperature", -1.0).toDouble();
    t = s.value("receiver_temperature_file", "").toByteArray();
    if (t.size() > 0)
    {
        tel->receiver_temperature_file = (char*)malloc(t.size() + 1);
        strcpy(tel->receiver_temperature_file, t.constData());
    }
    if (s.contains("receiver_temperature_file") &&
            s.contains("receiver_temperature"))
    {
        printf("== WARNING: Receiver temperature specified in two different "
                "ways, which may lead to unpredictable results!\n");
    }

    // Load MeerKAT element patterns.
//    element_pattern_meerkat_pol1_ = s.value("element_pattern_meerkat_pol1").toStringList();
//    element_pattern_meerkat_pol2_ = s.value("element_pattern_meerkat_pol2").toStringList();
//    element_pattern_files_meerkat_pol1_ = (char**)malloc(element_pattern_meerkat_pol1_.size() * sizeof(char*));
//    element_pattern_files_meerkat_pol2_ = (char**)malloc(element_pattern_meerkat_pol2_.size() * sizeof(char*));
//    for (int i = 0; i < element_pattern_meerkat_pol1_.size(); ++i)
//    {
//        int len = 1 + element_pattern_meerkat_pol1_[i].length();
//        element_pattern_files_meerkat_pol1_[i] = (char*)malloc(len);
//        t = element_pattern_meerkat_pol1_[i].toAscii();
//        strcpy(element_pattern_files_meerkat_pol1_[i], t);
//    }
//    for (int i = 0; i < element_pattern_meerkat_pol2_.size(); ++i)
//    {
//        int len = 1 + element_pattern_meerkat_pol2_[i].length();
//        element_pattern_files_meerkat_pol2_[i] = (char*)malloc(len);
//        t = element_pattern_meerkat_pol2_[i].toAscii();
//        strcpy(element_pattern_files_meerkat_pol2_[i], t);
//    }

    return 0;
}



// Code to free MeerKAT element pattern data.

//for (int i = 0; i < element_pattern_meerkat_pol1_.size(); ++i)
//    free(element_pattern_files_meerkat_pol1_[i]);
//free(element_pattern_files_meerkat_pol1_);
//for (int i = 0; i < element_pattern_meerkat_pol2_.size(); ++i)
//    free(element_pattern_files_meerkat_pol2_[i]);
//free(element_pattern_files_meerkat_pol2_);
