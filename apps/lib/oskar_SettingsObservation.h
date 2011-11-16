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

#include "interferometry/oskar_SimTime.h"
#include <QtCore/QString>
#include <QtCore/QSettings>

#define DEG2RAD 0.0174532925199432957692

class oskar_SettingsObservation
{
    public:
        void load(const QSettings& settings);

    public:
        double frequency(int channel) const
        { return start_frequency_ + channel * frequency_inc_; }

        double start_frequency() const { return start_frequency_; }
        void set_start_frequency(double value) { start_frequency_ = value; }

        int num_channels() const { return num_channels_; }
        void set_num_channels(int value) { num_channels_ = value; }

        double frequency_inc() const { return frequency_inc_; }
        void set_frequency_inc(double value) { frequency_inc_ = value; }

        double channel_bandwidth() const { return channel_bandwidth_; }
        void set_channel_bandwidth(double value) { channel_bandwidth_ = value; }

        double ra0_deg() const { return ra0_deg_; }
        void set_ra0_deg(double value) { ra0_deg_ = value; }
        double ra0_rad() const { return ra0_deg_ * DEG2RAD; }

        double dec0_deg() const { return dec0_deg_; }
        void set_dec0_deg(double value) { dec0_deg_ = value; }
        double dec0_rad() const { return dec0_deg_ * DEG2RAD; }

        QString oskar_vis_filename() const { return oskar_vis_filename_; }
        void set_oskar_vis_filename(const QString& value)
        { oskar_vis_filename_ = value; }

        QString ms_filename() const { return ms_filename_; }
        void set_ms_filename(const QString& value) { ms_filename_ = value; }

        const oskar_SimTime* sim_time() const {return &time_;}
        void set_sim_time(const oskar_SimTime& data) {time_ = data;}

    private:
        double   start_frequency_;
        int      num_channels_;
        double   frequency_inc_;
        double   channel_bandwidth_;
        double   ra0_deg_;
        double   dec0_deg_;

        QString  oskar_vis_filename_;
        QString  ms_filename_;

        oskar_SimTime time_;
};

#endif // OSKAR_SETTINGS_OBSERVATION_H_
