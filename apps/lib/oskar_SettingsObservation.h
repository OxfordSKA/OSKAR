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

#ifndef OSKAR_SETTINGS_OBSERVATION_H_
#define OSKAR_SETTINGS_OBSERVATION_H_

#include <QtCore/QString>
#include <QtCore/QSettings>

class oskar_SettingsObservation
{
    public:
        void load(const QSettings& settings);

    public:
        static const double deg2rad = 0.0174532925199432957692;

        double frequency(const unsigned channel) const
        { return _start_frequency + channel * _frequency_inc; }

        double start_frequency() const { return _start_frequency; }
        void set_start_frequency(const double value) { _start_frequency = value; }

        unsigned num_channels() const { return _num_channels; }
        void set_num_channels(const unsigned value) { _num_channels = value; }

        double frequency_inc() const { return _frequency_inc; }
        void set_frequency_inc(const double value) { _frequency_inc = value; }

        double channel_bandwidth() const { return _channel_bandwidth; }
        void set_channel_bandwidth(const double value) { _channel_bandwidth = value; }

        double ra0_deg() const { return _ra0_deg; }
        void set_ra0_deg(const double value) { _ra0_deg = value; }
        double ra0_rad() const { return _ra0_deg * deg2rad; }

        double dec0_deg() const { return _dec0_deg; }
        void set_dec0_deg(const double value) { _dec0_deg = value; }
        double dec0_rad() const { return _dec0_deg * deg2rad; }

        unsigned start_time_utc_year() const { return _start_time_utc_year; }
        void set_start_time_utc_year(const unsigned value) { _start_time_utc_year = value; }
        unsigned start_time_utc_month() const { return _start_time_utc_month; }
        void set_start_time_utc_month(const unsigned value) { _start_time_utc_month = value; }
        unsigned start_time_utc_day() const { return _start_time_utc_day; }
        void set_start_time_utc_day(const unsigned value) { _start_time_utc_day = value; }
        unsigned start_time_utc_hour() const { return _start_time_utc_hour; }
        void set_start_time_utc_hour(const unsigned value) { _start_time_utc_hour = value; }
        unsigned start_time_utc_minute() const { return _start_time_utc_minute; }
        void set_start_time_utc_minute(const unsigned value) { _start_time_utc_minute = value; }
        double start_time_utc_second() const { return _start_time_utc_second; }
        void set_start_time_utc_second(const double value) { _start_time_utc_second = value; }

        double start_time_utc_mjd() const { return _start_time_utc_mjd; }
        void set_start_time_utc_mjd(const double value) { _start_time_utc_mjd = value; }

        double obs_length_sec() const { return _obs_length_sec; }
        void set_obs_length_sec(const double value) { _obs_length_sec = value; }
        double obs_length_days() const
        { return _obs_length_sec / (24.0 * 60.0 * 60.0); }

        unsigned num_vis_dumps() const { return _num_vis_dumps; }
        void set_num_vis_dumps(const unsigned value) { _num_vis_dumps = value; }

        unsigned num_vis_ave() const { return _num_vis_ave; }
        void set_num_vis_ave(const unsigned value)  { _num_vis_ave = value; }

        unsigned num_fringe_ave() const { return _num_fringe_ave; }
        void set_num_fringe_ave(const unsigned value) { _num_fringe_ave = value; }

        QString oskar_vis_filename() const { return _oskar_vis_filename; }
        void set_oskar_vis_filename(const QString& value)
        { _oskar_vis_filename = value; }

        QString ms_filename() const { return _ms_filename; }
        void set_ms_filename(const QString& value) { _ms_filename = value; }

    private:
        double   _start_frequency;
        unsigned _num_channels;
        double   _frequency_inc;
        double   _channel_bandwidth;
        double   _ra0_deg;
        double   _dec0_deg;

        unsigned _start_time_utc_year;
        unsigned _start_time_utc_month;
        unsigned _start_time_utc_day;
        unsigned _start_time_utc_hour;
        unsigned _start_time_utc_minute;
        double   _start_time_utc_second;
        double   _start_time_utc_mjd;
        double   _obs_length_sec;

        unsigned _num_vis_dumps;
        unsigned _num_vis_ave;
        unsigned _num_fringe_ave;
        QString  _oskar_vis_filename;
        QString  _ms_filename;
};

#endif // OSKAR_SETTINGS_OBSERVATION_H_

