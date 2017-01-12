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

#include "oskar_settings_load_observation.h"

#include "log/oskar_log.h"

#include "math/oskar_cmath.h"
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QDateTime>
#include <QtCore/QVariant>
#include <QtCore/QDate>
#include <QtCore/QTime>
#include <QtCore/QString>
#include <QtCore/QStringList>

static double convert_date_time_to_mjd(int year, int month, int day,
        double day_fraction)
{
    int a, y, m, jdn;

    /* Compute Julian Day Number (Note: all integer division). */
    a = (14 - month) / 12;
    y = year + 4800 - a;
    m = month + 12 * a - 3;
    jdn = day + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
            + (y / 400) - 32045;

    /* Compute day fraction. */
    day_fraction -= 0.5;
    return (jdn - 2400000.5) + day_fraction;
}


extern "C"
void oskar_settings_load_observation(oskar_SettingsObservation* obs,
        oskar_Log* log, const char* filename, int* status)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    if (*status) return;

    s.beginGroup("observation");
    {
        // Get observation start time as MJD(UTC).
        QString str_st = s.value("start_time_utc").toString();
        if (str_st.size() > 0 && !str_st.contains(":"))
        {
            bool ok = false;
            obs->start_mjd_utc = str_st.toDouble(&ok);
            if (!ok)
            {
                oskar_log_error(log, "Invalid string for 'start_time_utc'.");
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }
        }
        else
        {
            const char* format_strings[] = {
                    "d-M-yyyy h:m:s.z", // British style
                    "d-M-yyyy h:m:s",
                    "yyyy/M/d/h:m:s.z", // CASA style
                    "yyyy/M/d/h:m:s",
                    "yyyy-MM-dd h:m:s.z", // International style
                    "yyyy-MM-dd h:m:s",
                    "yyyy-MM-ddTh:m:s.z", // ISO date style
                    "yyyy-MM-ddTh:m:s",
            };
            QDateTime st;
            int i = 0, num_formats = sizeof(format_strings) / sizeof(char*);
            for (i = 0; i < num_formats; ++i)
            {
                st = QDateTime::fromString(str_st, format_strings[i]);
                if (st.isValid())
                    break;
            }
            if (i == num_formats)
            {
                oskar_log_error(log, "Invalid string for 'start_time_utc'.");
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }
            int year   = st.date().year();
            int month  = st.date().month();
            int day    = st.date().day();
            int hour   = st.time().hour();
            int minute = st.time().minute();
            double second = st.time().second() + st.time().msec() / 1000.0;

            // Compute start time as MJD(UTC).
            double day_fraction = (hour + (minute / 60.0) +
                    (second / 3600.0)) / 24.0;
            obs->start_mjd_utc = convert_date_time_to_mjd(year,
                    month, day, day_fraction);
        }

        // Get number of time steps.
        obs->num_time_steps  = s.value("num_time_steps", 1).toInt();

        // Get observation length, either as number of seconds, or as a string.
        QString length = s.value("length").toString();
        if (length.size() > 0 && !length.contains(":"))
        {
            bool ok = false;
            obs->length_sec = length.toDouble(&ok);
            if (!ok)
            {
                oskar_log_error(log, "Invalid time string for 'length'");
                *status = OSKAR_ERR_INVALID_ARGUMENT;
                return;
            }
        }
        else
        {
            QTime len = QTime::fromString(length, "h:m:s.z");
            if (!len.isValid())
            {
                len = QTime::fromString(length, "h:m:s");
                if (!len.isValid())
                {
                    oskar_log_error(log, "Invalid time string for 'length' "
                            "(format must be hh:mm:ss.z).");
                    *status = OSKAR_ERR_INVALID_ARGUMENT;
                    return;
                }
            }
            obs->length_sec = len.hour() * 3600.0 +
                    len.minute() * 60.0 + len.second() + len.msec() / 1000.0;
        }
        obs->length_days = obs->length_sec / 86400.0;
    }
    s.endGroup();

    // Range checks.
    if (obs->num_time_steps <= 0) obs->num_time_steps = 1;

    // Compute interval
    obs->dt_dump_days = obs->length_days / obs->num_time_steps;
}
