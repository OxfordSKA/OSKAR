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

#include <cstdio>
#include <cstdlib>


void oskar_SettingsObservation::load(const QSettings& settings)
{
    _start_frequency   = settings.value("observation/start_frequency").toDouble();
    _num_channels      = settings.value("observation/num_channels").toUInt();
    _frequency_inc     = settings.value("observation/frequency_inc").toDouble();
    _channel_bandwidth = settings.value("observation/channel_bandwidth").toDouble();
    _ra0_deg           = settings.value("observation/phase_centre_ra_deg").toDouble();
    _dec0_deg          = settings.value("observation/phase_centre_dec_deg").toDouble();

    _start_time_utc_year   = settings.value("observation/start_time_utc_year").toUInt();
    _start_time_utc_month  = settings.value("observation/start_time_utc_month").toUInt();
    _start_time_utc_day    = settings.value("observation/start_time_utc_day").toUInt();
    _start_time_utc_hour   = settings.value("observation/start_time_utc_hour").toUInt();
    _start_time_utc_minute = settings.value("observation/start_time_utc_minute").toUInt();
    _start_time_utc_second = settings.value("observation/start_time_utc_second").toDouble();

    double day_fraction = (_start_time_utc_hour +
            (_start_time_utc_minute / 60.0) +
            (_start_time_utc_second / 3600.0)) / 24.0;
    _start_time_utc_mjd = oskar_date_time_to_mjd(_start_time_utc_year,
            _start_time_utc_month, _start_time_utc_day,
            day_fraction);

    _obs_length_sec    = settings.value("observation/length_seconds").toDouble();
    _oskar_vis_filename= settings.value("observation/oskar_vis_filename", "").toString();
    _ms_filename       = settings.value("observation/ms_filename", "").toString();
    _num_vis_dumps     = settings.value("observation/num_vis_dumps").toUInt();
    _num_vis_ave       = settings.value("observation/num_vis_ave").toUInt();
    _num_fringe_ave    = settings.value("observation/num_fringe_ave").toUInt();
}
