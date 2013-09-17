/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "apps/lib/oskar_settings_load_interferometer.h"
#include "sky/oskar_date_time_to_mjd.h"
#include <oskar_log.h>

#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QDateTime>
#include <QtCore/QVariant>
#include <QtCore/QDate>
#include <QtCore/QTime>
#include <QtCore/QString>

#include <cmath>
#include <cstdlib>
#include <cstring>

// local (private) methods
static int load_noise(oskar_SettingsSystemNoise* noise, const char* filename);
static int load_noise_freqs(oskar_SettingsSystemNoiseFreq* freq, QSettings& s);
static int load_noise_values(oskar_SettingsSystemNoiseValue* value, QSettings& s);
static int load_noise_type(oskar_SettingsSystemNoiseType* stddev, QSettings& s,
        const QString& name);
static void get_filename(char** filename, const QSettings& s, const QString& key,
        const QString& defaultValue = "");


extern "C"
int oskar_settings_load_interferometer(oskar_SettingsInterferometer* settings,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    s.beginGroup("interferometer");
    {
        settings->channel_bandwidth_hz = s.value("channel_bandwidth_hz").toDouble();
        settings->time_average_sec = s.value("time_average_sec", 0.0).toDouble();
        settings->num_vis_ave      = s.value("num_vis_ave", 1).toInt();
        settings->num_fringe_ave   = s.value("num_fringe_ave", 1).toInt();
        settings->use_common_sky   = s.value("use_common_sky", true).toBool();

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

    int status = load_noise(&settings->noise, filename);
    if (status) return status;

    return OSKAR_SUCCESS;
}


static int load_noise(oskar_SettingsSystemNoise* noise, const char* filename)
{
    QByteArray t;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);
    int error = OSKAR_SUCCESS;

    s.beginGroup("interferometer");
    {
        s.beginGroup("noise");
        {
            /* Enable / disable */
            noise->enable = s.value("enable", false).toBool();

            /* Seed */
            temp = s.value("seed").toString().toUpper();
            noise->seed = (temp == "TIME" || temp.toInt() < 0) ?
                    (int)time(NULL) : temp.toInt();

            /* Load frequency settings */
            error = load_noise_freqs(&noise->freq, s);
            if (error) return error;

            /* Values */
            error = load_noise_values(&noise->value, s);
            if (error) return error;
        }
        s.endGroup(); // noise
    }
    s.endGroup(); // interferometer

    return error;
}

static int load_noise_freqs(oskar_SettingsSystemNoiseFreq* freq,
        QSettings& s)
{
    QString temp = s.value("freq", "T").toString().toUpper();
    if (temp.startsWith("T"))
        freq->specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    else if (temp.startsWith("O"))
        freq->specification = OSKAR_SYSTEM_NOISE_OBS_SETTINGS;
    else if (temp.startsWith("D"))
        freq->specification = OSKAR_SYSTEM_NOISE_DATA_FILE;
    else if (temp.startsWith("R"))
        freq->specification = OSKAR_SYSTEM_NOISE_RANGE;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup("freq");
    {
        get_filename(&freq->file, s, "file");
        freq->number = s.value("number").toInt();
        freq->start  = s.value("start").toDouble();
        freq->inc    = s.value("inc").toDouble();
    }
    s.endGroup(); // freq

    return OSKAR_SUCCESS;
}


static int load_noise_values(oskar_SettingsSystemNoiseValue* value,
        QSettings& s)
{
    int error = OSKAR_SUCCESS;

    QString temp = s.value("values", "TEL").toString().toUpper();
    if (temp.startsWith("TEL"))
        value->specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    else if (temp.startsWith("R"))
        value->specification = OSKAR_SYSTEM_NOISE_RMS;
    else if (temp.startsWith("S"))
        value->specification = OSKAR_SYSTEM_NOISE_SENSITIVITY;
    else if(temp.startsWith("TEMP"))
        value->specification = OSKAR_SYSTEM_NOISE_SYS_TEMP;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup("values");
    {
        error = load_noise_type(&value->rms, s, "rms");
        if (error) return error;

        error = load_noise_type(&value->sensitivity, s, "sensitivity");
        if (error) return error;

        s.beginGroup("components");
        {
            error = load_noise_type(&value->t_sys, s, "t_sys");
            if (error) return error;

            error = load_noise_type(&value->area, s, "area");
            if (error) return error;

            error = load_noise_type(&value->efficiency, s, "efficiency");
            if (value->efficiency.start < 0.0 || value->efficiency.start > 1.0)
                return OSKAR_ERR_SETTINGS_INTERFEROMETER;
            if (value->efficiency.end < 0.0 || value->efficiency.end > 1.0)
                 return OSKAR_ERR_SETTINGS_INTERFEROMETER;
            if (value->efficiency.start > value->efficiency.end)
                return OSKAR_ERR_SETTINGS_INTERFEROMETER;
            if (error) return error;
        }
        s.endGroup();
    }
    s.endGroup(); // spec

    return error;
}


static int load_noise_type(oskar_SettingsSystemNoiseType* type, QSettings& s,
        const QString& name)
{
    QString temp = s.value(name, "N").toString().toUpper();
    if (temp.startsWith("N"))
        type->override = OSKAR_SYSTEM_NOISE_NO_OVERRIDE;
    else if (temp.startsWith("D"))
        type->override = OSKAR_SYSTEM_NOISE_DATA_FILE;
    else if (temp.startsWith("R"))
        type->override = OSKAR_SYSTEM_NOISE_RANGE;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup(name);
    {
        get_filename(&type->file, s, "file");
        type->start  = s.value("start").toDouble();
        type->end    = s.value("end").toDouble();
    }
    s.endGroup();

    return OSKAR_SUCCESS;
}


static void get_filename(char** filename, const QSettings& s,
        const QString& key, const QString& defaultValue)
{
    QByteArray t = s.value(key, defaultValue).toByteArray();
    if (t.size() > 0)
    {
        *filename = (char*)malloc(t.size() + 1);
        strcpy(*filename, t.constData());
    }
}
