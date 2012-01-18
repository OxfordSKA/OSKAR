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
#include "apps/lib/oskar_SettingsBenchmark.h"
#include "apps/lib/oskar_SettingsSky.h"

#include <QtCore/QString>
#include <QtCore/QStringList>
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
        QString telescope_file() const { return telescope_file_; }
        void set_telescope_file(const QString& value) { telescope_file_ = value; }

        double longitude_deg() const { return longitude_deg_; }
        void set_longitude_deg(const double value) { longitude_deg_ = value; }
        double latitude_deg() const { return latitude_deg_; }
        void set_latitude_deg(const double value) { latitude_deg_ = value; }
        double longitude_rad() const { return longitude_deg_ * DEG2RAD; }
        double latitude_rad() const { return latitude_deg_ * DEG2RAD; }
        double altitude_m() const { return altitude_m_; }
        void set_altitude_m(const double value) { altitude_m_ = value; }

        QString station_dir() const { return station_dir_; }
        void set_station_dir(const QString& value) { station_dir_ = value; }
        bool disable_station_beam() const { return disable_station_beam_; }
        void set_disable_station_beam(const bool value)
        { disable_station_beam_ = value; }

        bool double_precision() const { return prec_double_; }
        int max_sources_per_chunk() const { return max_sources_per_chunk_; }
        int num_devices() const { return use_devices_.size(); }
        const int* use_devices() const { return use_devices_.constData(); }
        const char* const* element_pattern_files_meerkat_pol1() const {return element_pattern_files_meerkat_pol1_;}
        const char* const* element_pattern_files_meerkat_pol2() const {return element_pattern_files_meerkat_pol2_;}
        int num_element_pattern_files_meerkat_pol1() const {return element_pattern_meerkat_pol1_.size();}
        int num_element_pattern_files_meerkat_pol2() const {return element_pattern_meerkat_pol2_.size();}

        const oskar_SettingsObservation& obs() const { return obs_; }
        oskar_SettingsObservation& obs() { return obs_; }

        const oskar_SettingsImage& image() const { return image_; }
        oskar_SettingsImage& image() { return image_; }

        const oskar_SettingsBenchmark& benchmark() const { return benchmark_; }
        const oskar_SettingsSky& sky() const { return sky_; }
        oskar_SettingsSky& sky() { return sky_; }

    private:
        bool prec_double_;
        int max_sources_per_chunk_;
        QVector<int> use_devices_;
        QString filename_;
        QString telescope_file_;
        double longitude_deg_;
        double latitude_deg_;
        double altitude_m_;
        QString station_dir_;
        bool disable_station_beam_;
        QStringList element_pattern_meerkat_pol1_;
        QStringList element_pattern_meerkat_pol2_;
        oskar_SettingsObservation obs_;
        oskar_SettingsImage image_;
        oskar_SettingsBenchmark benchmark_;
        oskar_SettingsSky sky_;
        char** element_pattern_files_meerkat_pol1_;
        char** element_pattern_files_meerkat_pol2_;
};

#endif // OSKAR_SETTINGS_H_
