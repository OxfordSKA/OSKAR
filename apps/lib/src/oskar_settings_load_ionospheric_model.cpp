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


#include "apps/lib/oskar_settings_load_ionospheric_model.h"

extern "C"
int oskar_settings_load_ionospheric_model(oskar_SettingsMIM* mim,
        const char* filename)
{
    QString temp;
    QByteArray t;

    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("ionosphere");

    mim->enable = (int)s.value("enable", false).toBool();
    mim->height_km = (double)s.value("height_km", 300.0).toDouble();

    s.beginGroup("TID");
    mim->enableTID = (int)s.value("enable", true).toBool();

    temp = s.value("config_type", "Component model").toString().toUpper();
    if (temp.startsWith("COMP"))
    {
        mim->num_tid_components = 2;
        mim->tid = (oskar_SettingsTID*) malloc(
                sizeof(oskar_SettingsTID) * mim->num_tid_components);
        s.beginGroup("components");
        int g = 0;
        s.beginGroup("0");
        mim->tid[g].amp = s.value("amp", 0.0).toDouble();
        mim->tid[g].theta = s.value("theta", 0.0).toDouble();
        mim->tid[g].speed = s.value("speed", 0.0).toDouble();
        mim->tid[g].wavelength = s.value("wavelength", 0.0).toDouble();
        s.endGroup(); // 0
        g = 1;
        s.beginGroup("1");
        mim->tid[g].amp = s.value("amp", 0.0).toDouble();
        mim->tid[g].theta = s.value("theta", 0.0).toDouble();
        mim->tid[g].speed = s.value("speed", 0.0).toDouble();
        mim->tid[g].wavelength = s.value("wavelength", 0.0).toDouble();
        s.endGroup(); // 1
        s.endGroup(); // components

    }
    else if (temp.startsWith("CONF"))
    {
        t = s.value("").toByteArray();
        if (t.size() > 0)
        {
            mim->component_file = (char*)malloc(t.size() + 1);
            strcpy(mim->component_file, t.constData());
        }
        else
            return OSKAR_ERR_SETTINGS_IONOSPHERE;

        // TODO read the component file!
    }
    else
    {
        return OSKAR_ERR_SETTINGS_IONOSPHERE;
    }

    s.endGroup(); // TID


    s.beginGroup("output");

    s.endGroup();

    s.endGroup(); // ionosphere

    return OSKAR_SUCCESS;
}


