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

#include "apps/lib/oskar_settings_load_observation.h"
#include "sky/oskar_date_time_to_mjd.h"

#include <cmath>
#include <cstdio>
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
int oskar_settings_load_observation(oskar_SettingsObservation* obs,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);
    s.beginGroup("observation");

    // Get frequency / channel data.
    obs->num_channels         = s.value("num_channels", 1).toInt();
    obs->start_frequency_hz   = s.value("start_frequency_hz").toDouble();
    obs->frequency_inc_hz     = s.value("frequency_inc_hz").toDouble();
    obs->channel_bandwidth_hz = s.value("channel_bandwidth_hz").toDouble();

    // Get pointing direction.
    obs->ra0_rad = s.value("phase_centre_ra_deg").toDouble() * M_PI / 180.0;
    obs->dec0_rad = s.value("phase_centre_dec_deg").toDouble() * M_PI / 180.0;

    // Get time data.
    obs->time.num_vis_dumps  = s.value("num_vis_dumps", 1).toInt();
    obs->time.num_vis_ave    = s.value("num_vis_ave", 1).toInt();
    obs->time.num_fringe_ave = s.value("num_fringe_ave", 1).toInt();

    // Get observation start time (if blank, then use current).
    QString str_st = s.value("start_time_utc").toString();
    QDateTime st = (str_st.isEmpty()) ?  QDateTime::currentDateTime().toUTC()
            : QDateTime::fromString(str_st, "d-M-yyyy h:m:s.z");
    if (!st.isValid())
    {
        fprintf(stderr, "ERROR: Invalid date string for 'start_time_utc' "
                "(format must be: 'd-M-yyyy h:m:s.z').\n");
        return OSKAR_ERR_SETTINGS;
    }
    int year   = st.date().year();
    int month  = st.date().month();
    int day    = st.date().day();
    int hour   = st.time().hour();
    int minute = st.time().minute();
    double second = st.time().second() + st.time().msec() / 1000.0;

    // Compute start time as MJD(UTC).
    double day_fraction = (hour + (minute / 60.0) + (second / 3600.0)) / 24.0;
    obs->time.obs_start_mjd_utc = oskar_date_time_to_mjd(year, month, day,
            day_fraction);

    // Get observation length.
    QString str_len = s.value("length").toString();
    if (str_len.isEmpty()) str_st = "00:00:01.000";
    QTime len = QTime::fromString(str_len, "h:m:s.z");
    if (!len.isValid())
    {
        fprintf(stderr, "ERROR: Invalid time string for 'length' "
                "(format must be: 'h:m:s.z').\n");
        return OSKAR_ERR_SETTINGS;
    }
    obs->time.obs_length_seconds = len.hour() * 3600.0 +
            len.minute() * 60.0 + len.second() + len.msec() / 1000.0;
    obs->time.obs_length_days = obs->time.obs_length_seconds / 86400.0;

    // Compute intervals.
    obs->time.dt_dump_days = obs->time.obs_length_days / obs->time.num_vis_dumps;
    obs->time.dt_ave_days = obs->time.dt_dump_days / obs->time.num_vis_ave;
    obs->time.dt_fringe_days = obs->time.dt_ave_days / obs->time.num_fringe_ave;

    // Get output visibility file name.
    t = s.value("oskar_vis_filename", "").toByteArray();
    if (t.size() > 0)
    {
        obs->oskar_vis_filename = (char*)malloc(t.size() + 1);
        strcpy(obs->oskar_vis_filename, t.constData());
    }

    // Get output MS file name.
    t = s.value("ms_filename", "").toByteArray();
    if (t.size() > 0)
    {
        obs->ms_filename = (char*)malloc(t.size() + 1);
        strcpy(obs->ms_filename, t.constData());
    }

    return 0;
}
