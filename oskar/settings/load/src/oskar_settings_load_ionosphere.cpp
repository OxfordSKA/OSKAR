/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_settings_load_ionosphere.h>
#include <oskar_settings_load_tid_parameter_file.h>

#include <cstring>
#include <cstdlib>

#include <QtCore/QString>
#include <QtCore/QByteArray>
#include <QtCore/QSettings>
#include <QtCore/QStringList>

#include <oskar_cmath.h>

extern "C"
void oskar_settings_load_ionosphere(oskar_SettingsIonosphere* settings,
        const char* filename, int* status)
{
    QString temp;
    QByteArray t;
    QStringList list;
    QSettings s(QString(filename), QSettings::IniFormat);

    // Check if safe to proceed.
    if (*status) return;

    s.beginGroup("ionosphere");
    settings->enable = (int)s.value("enable", false).toBool();
    settings->min_elevation = (double)s.value("min_elevation_deg", 0.0).toDouble();
    settings->min_elevation *= M_PI/180.0; // Convert to radians.

    settings->TEC0 = (double)s.value("TEC0", 1.0).toDouble();
    list = s.value("TID_file").toStringList();
    settings->num_TID_screens = list.size();
    settings->TID_files = (char**)malloc(settings->num_TID_screens*sizeof(char*));
    settings->TID = (oskar_SettingsTIDscreen*)malloc(
            settings->num_TID_screens*sizeof(oskar_SettingsTIDscreen));
    for (int i = 0; i < settings->num_TID_screens; ++i)
    {
        t = list[i].toLatin1();
        settings->TID_files[i] = (char*)malloc(t.size() + 1);
        strcpy(settings->TID_files[i], t.constData());
        oskar_settings_load_tid_parameter_file(&settings->TID[i],
                settings->TID_files[i], status);
    }

    s.beginGroup("TECImage");
    settings->TECImage.beam_centred = s.value("beam_centred", true).toBool();
    settings->TECImage.stationID = s.value("station_idx", 0).toInt();
    settings->TECImage.fov_rad = s.value("fov_deg", 0.0).toDouble();
    settings->TECImage.fov_rad *= (M_PI/180.);
    settings->TECImage.size = s.value("size", 0).toInt();
    temp = s.value("filename").toString();
    temp += "_st_" + QString::number(settings->TECImage.stationID);
    temp += (settings->TECImage.beam_centred) ? "_BC" : "_SC";
    if (s.value("save_fits", true).toBool())
    {
        QString temp_fits = temp + ".fits";
        t = temp_fits.toLatin1();
        settings->TECImage.fits_file = (char*)malloc(t.size() + 1);
        strcpy(settings->TECImage.fits_file, t.constData());
    }
    s.endGroup(); // TECImage

    s.beginGroup("pierce_points");
    temp = s.value("filename").toString();
    t = temp.toLatin1();
    settings->pierce_points.filename = (char*)malloc(t.size() + 1);
    strcpy(settings->pierce_points.filename, t.constData());
    s.endGroup(); // pierce_points

    s.endGroup(); // ionosphere
}
