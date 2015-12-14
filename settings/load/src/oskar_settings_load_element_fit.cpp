/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_settings_load_element_fit.h>
#include <oskar_station.h>

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

#define D2R (M_PI/180.0)

extern "C"
void oskar_settings_load_element_fit(oskar_SettingsElementFit* settings,
        const char* filename, int* status)
{
    QString temp;
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    if (*status) return;

    s.beginGroup("element_fit");
    t = s.value("input_cst_file").toByteArray();
    if (t.size() > 0)
    {
        settings->input_cst_file = (char*)malloc(t.size() + 1);
        strcpy(settings->input_cst_file, t.constData());
    }
    t = s.value("input_scalar_file").toByteArray();
    if (t.size() > 0)
    {
        settings->input_scalar_file = (char*)malloc(t.size() + 1);
        strcpy(settings->input_scalar_file, t.constData());
    }
    t = s.value("output_directory").toByteArray();
    if (t.size() > 0)
    {
        settings->output_directory = (char*)malloc(t.size() + 1);
        strcpy(settings->output_directory, t.constData());
    }
    t = s.value("fits_image").toByteArray();
    if (t.size() > 0)
    {
        settings->fits_image = (char*)malloc(t.size() + 1);
        strcpy(settings->fits_image, t.constData());
    }
    t = s.value("pol_type", "XY").toByteArray().toUpper();
    if (t == "X")
    {
        settings->pol_type = 1;
    }
    else if (t == "Y")
    {
        settings->pol_type = 2;
    }
    else if (t == "XY")
    {
        settings->pol_type = 0;
    }
    else
    {
        *status = OSKAR_ERR_SETTINGS;
        return;
    }

    settings->element_type_index = s.value("element_type_index").toInt();
    settings->frequency_hz = s.value("frequency_hz").toDouble();
    settings->ignore_data_at_pole =
            s.value("ignore_data_at_pole", false).toBool();
    settings->ignore_data_below_horizon =
            s.value("ignore_data_below_horizon", true).toBool();
    settings->average_fractional_error =
            s.value("average_fractional_error", 0.005).toDouble();
    settings->average_fractional_error_factor_increase =
            s.value("average_fractional_error_factor_increase", 1.1).toDouble();
    s.endGroup();
}
