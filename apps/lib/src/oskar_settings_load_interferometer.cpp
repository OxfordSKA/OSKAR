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

#include "apps/lib/oskar_settings_load_interferomter.h"
#include "sky/oskar_date_time_to_mjd.h"
#include "utility/oskar_log_error.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QDateTime>
#include <QtCore/QVariant>
#include <QtCore/QDate>
#include <QtCore/QTime>
#include <QtCore/QString>

extern "C"
int oskar_settings_load_interferometer(oskar_SettingsInterferometer* settings,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    s.beginGroup("interferometer");
    {
        settings->channel_bandwidth_hz = s.value("channel_bandwidth_hz").toDouble();
        settings->num_vis_ave     = s.value("num_vis_ave", 1).toInt();
        settings->num_fringe_ave  = s.value("num_fringe_ave", 1).toInt();

        // Get output visibility file name.
        t = s.value("oskar_vis_filename", "").toByteArray();
        if (t.size() > 0)
        {
            settings->oskar_vis_filename = (char*)malloc(t.size() + 1);
            strcpy(settings->oskar_vis_filename, t.constData());
        }

        // Get output MS file name.
        t = s.value("ms_filename", "").toByteArray();
        if (t.size() > 0)
        {
            settings->ms_filename = (char*)malloc(t.size() + 1);
            strcpy(settings->ms_filename, t.constData());
        }

        settings->image_interferometer_output = s.value("image_output", false).toBool();
    }
    s.endGroup();

    // Range checks.
    if (settings->num_vis_ave <= 0) settings->num_vis_ave = 1;
    if (settings->num_fringe_ave <= 0) settings->num_fringe_ave = 1;

    return OSKAR_SUCCESS;
}
