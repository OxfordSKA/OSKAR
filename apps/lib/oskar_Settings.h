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

#include "apps/lib/oskar_SettingsImage.h"
#include "apps/lib/oskar_SettingsObservation.h"

#include <QtCore/QString>
#include <QtCore/QSettings>

#define DEG2RAD 0.0174532925199432957692

class oskar_Settings
{
    public:
        oskar_Settings(const QString& filename = QString());
        ~oskar_Settings();

    public:
        int load(const QString& filename = QString());

        int check() const;

        void print() const;

    public:
        QString sky_file() const { return _sky_file; }
        void set_sky_file(const QString& value) { _sky_file = value; }

        QString telescope_file() const { return _telescope_file; }
        void set_telescope_file(const QString& value) { _telescope_file = value; }

        double longitude_deg() const { return _longitude_deg; }
        void set_longitude_deg(const double value) { _longitude_deg = value; }
        double latitude_deg() const { return _latitude_deg; }
        void set_latitude_deg(const double value) { _latitude_deg = value; }
        double longitude_rad() const { return _longitude_deg * DEG2RAD; }
        double latitude_rad() const { return _latitude_deg * DEG2RAD; }
        double altitude_m() const { return _altitude_m; }
        void set_altitude_m(const double value) { _altitude_m = value; }

        QString station_dir() const { return _station_dir; }
        void set_station_dir(const QString& value) { _station_dir = value; }
        bool disable_station_beam() const { return _disable_station_beam; }
        void set_disable_station_beam(const bool value)
        { _disable_station_beam = value; }

        bool double_precision() const { return _prec_double; }

        const oskar_SettingsObservation& obs() const { return _obs; }
        oskar_SettingsObservation& obs() { return _obs; }

        const oskar_SettingsImage& image() const { return _image; }
        oskar_SettingsImage& image() { return _image; }

    private:
        bool _prec_double;

        QString _filename;

        QString _sky_file;

        QString _telescope_file;
        double _longitude_deg;
        double _latitude_deg;
        double _altitude_m;

        QString _station_dir;
        bool _disable_station_beam;

        oskar_SettingsObservation _obs;

        oskar_SettingsImage _image;
};

#endif // OSKAR_SETTINGS_H_

