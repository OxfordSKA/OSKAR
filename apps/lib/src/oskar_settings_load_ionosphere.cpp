/*
 * Copyright (c) 2013, The University of Oxford
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


#include "apps/lib/oskar_settings_load_ionosphere.h"
#include "sky/oskar_load_TID_parameter_file.h"

#include <QtCore/QString>
#include <QtCore/QByteArray>
#include <QtCore/QSettings>
#include <QtCore/QStringList>

extern "C"
int oskar_settings_load_ionosphere(oskar_SettingsIonosphere* settings,
        const char* filename)
{
    QString temp;
    QByteArray t;
    QStringList list;
    int status = OSKAR_SUCCESS;

    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("ionosphere");
    settings->enable = (int)s.value("enable", true).toBool();
    settings->min_elevation = (double)s.value("min_elevation_deg", 0.0).toDouble();
    settings->TEC0 = (double)s.value("TEC0", 1.0).toDouble();
    list = s.value("TID_file").toStringList();
    settings->num_TID_screens = list.size();
    settings->TID_files = (char**)malloc(settings->num_TID_screens*sizeof(char*));
    settings->TID = (oskar_SettingsTIDscreen*)malloc(
            settings->num_TID_screens*sizeof(oskar_SettingsTIDscreen));
    for (int i = 0; i < settings->num_TID_screens; ++i)
    {
        t = list[i].toAscii();
        settings->TID_files[i] = (char*)malloc(t.size() + 1);
        strcpy(settings->TID_files[i], t.constData());
        oskar_load_TID_parameter_file(&settings->TID[i], settings->TID_files[i],
                &status);
    }
    s.endGroup(); // ionosphere

    return status;
}
