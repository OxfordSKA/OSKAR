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

#ifndef OSKAR_SETTINGS_H_
#define OSKAR_SETTINGS_H_

#include <QtCore/QString>


class oskar_Settings
{
    public:
        oskar_Settings();
        ~oskar_Settings();

    public:
        int load(const QString& filename = QString());

        int check();

        void print();

    public:
        static const double deg2rad = 0.0174532925199432957692;

        QString sky_file() const { return _sky_file; }

        QString telescope_file() const { return _telescope_file; }
        double longitude_deg() const { return _longitude_deg; }
        double latitude_deg() const { return _latitude_deg; }
        double longitude_rad() const { return _longitude_deg * deg2rad; }
        double latitude_rad() const { return _latitude_deg * deg2rad; }

        QString station_dir() const { return _station_dir; }

        double frequency() const { return _frequency; }
        double channel_bandwidth() const { return _channel_bandwidth; }
        double ra0_deg() const { return _ra0_deg; }
        double dec0_deg() const { return _dec0_deg; }
        double ra0_rad() const { return _ra0_deg * deg2rad; }
        double dec0_rad() const { return _dec0_deg * deg2rad; }

        double obs_length_sec() const { return _obs_length_sec; }
        double obs_length_days() const
        { return _obs_length_sec / (24.0 * 60.0 * 60.0); }
        double obs_start_mjc_utc() const { return _obs_start_mjd_utc; }

        unsigned num_vis_dumps() const { return _num_vis_dumps; }
        unsigned num_vis_ave() const { return _num_vis_ave; }
        unsigned num_fringe_ave() const { return _num_fringe_ave; }

        QString output_file() const { return _output_file; }

    private:
        QString _filename;

        QString _sky_file;

        QString _telescope_file;
        double _longitude_deg;
        double _latitude_deg;

        QString _station_dir;

        double _frequency;
        double _channel_bandwidth;
        double _ra0_deg;
        double _dec0_deg;
        double _obs_length_sec;
        double _obs_start_mjd_utc;

        unsigned _num_vis_dumps;
        unsigned _num_vis_ave;
        unsigned _num_fringe_ave;

        QString _output_file;
};

#endif // OSKAR_SETTINGS_H_

