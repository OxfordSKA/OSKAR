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

#include "apps/lib/oskar_SettingsObservation.h"
#include "sky/oskar_date_time_to_mjd.h"

#include <QtCore/QFileInfo>
#include <QtCore/QSettings>
#include <QtCore/QStringList>
#include <QtCore/QDateTime>
#include <QtCore/QVariant>
#include <QtCore/QDate>
#include <QtCore/QTime>
#include <QtCore/QChar>
#include <QtCore/QString>

#include <cstdio>
#include <cstdlib>


void oskar_SettingsObservation::load(const QSettings& settings)
{
    // Get frequency / channel data.
    start_frequency_   = settings.value("observation/start_frequency").toDouble();
    num_channels_      = settings.value("observation/num_channels").toInt();
    frequency_inc_     = settings.value("observation/frequency_inc").toDouble();
    channel_bandwidth_ = settings.value("observation/channel_bandwidth").toDouble();

    // Get pointing direction.
//    QVariant ra0 = settings.value("observation/phase_centre_ra_deg");
//    printf("%s\n", ra0.typeName());
//    if (ra0.type() == QVariant::Double)
//    {
        ra0_deg_ = settings.value("observation/phase_centre_ra_deg").toDouble();
//    }
//    else if (ra0.type() == QVariant::String)
//    {
//        QStringList s_ra0 = ra0.toString().split(":");
//        if (s_ra0.length() != 3)
//        {
//            fprintf(stderr, "ERROR: invalid ra0 string. Required format = "
//                    "'deg:arcmin:arcsec'\n");
//            return;
//        }
//        ra0_deg_ = s_ra0.at(0).toInt() + s_ra0.at(1).toInt()/60.0 +
//                s_ra0.at(2).toDouble()/3600.0;
//    }
//    else
//    {
//        fprintf(stderr, "ERROR: observation/phase_centre_ra_deg => bad type!");
//    }

    dec0_deg_ = settings.value("observation/phase_centre_dec_deg").toDouble();

    // Get time data.
    time_.num_vis_dumps        = settings.value("observation/num_vis_dumps").toInt();
    time_.num_vis_ave          = settings.value("observation/num_vis_ave").toInt();
    time_.num_fringe_ave       = settings.value("observation/num_fringe_ave").toInt();

    // Get observation start time.
    QString str_st = settings.value("observation/start_time_utc").toString();
    QDateTime st = QDateTime::fromString(str_st, "d-M-yyyy h:m:s.z");
    if (!st.isValid())
    {
        fprintf(stderr, "ERROR: Invalid date string for 'start_time_utc' "
                "(format must be: 'd-M-yyyy h:m:s.z').\n");
    }
    time_.obs_start_utc_year   = st.date().year();
    time_.obs_start_utc_month  = st.date().month();
    time_.obs_start_utc_day    = st.date().day();
    time_.obs_start_utc_hour   = st.time().hour();
    time_.obs_start_utc_minute = st.time().minute();
    time_.obs_start_utc_second = st.time().second() + st.time().msec()/1000.0;

    // Compute start time as MJD(UTC).
    double day_fraction = (time_.obs_start_utc_hour +
            (time_.obs_start_utc_minute / 60.0) +
            (time_.obs_start_utc_second / 3600.0)) / 24.0;
    time_.obs_start_mjd_utc = oskar_date_time_to_mjd(time_.obs_start_utc_year,
            time_.obs_start_utc_month, time_.obs_start_utc_day, day_fraction);

    // Get observation length.
    QString str_len = settings.value("observation/length").toString();
    QTime len = QTime::fromString(str_len, "h:m:s.z");
    if (!len.isValid())
    {
        fprintf(stderr, "ERROR: Invalid time string for 'length' "
                "(format must be: 'h:m:s.z').\n");
    }
    time_.obs_length_seconds = len.hour() * 3600.0 + len.minute() * 60.0 +
            len.second() + len.msec() / 1000.0;
    time_.obs_length_days    = time_.obs_length_seconds / 86400.0;

    // Compute intervals.
    time_.dt_dump_days         = time_.obs_length_days / time_.num_vis_dumps;
    time_.dt_ave_days          = time_.dt_dump_days / time_.num_vis_ave;
    time_.dt_fringe_days       = time_.dt_ave_days / time_.num_fringe_ave;

    oskar_vis_filename_        = settings.value("observation/oskar_vis_filename", "").toString();
    ms_filename_               = settings.value("observation/ms_filename", "").toString();
}
