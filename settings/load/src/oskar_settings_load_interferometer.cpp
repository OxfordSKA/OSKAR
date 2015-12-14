/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_settings_load_interferometer.h>
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
#include <ctime>


static void load_noise(oskar_SettingsSystemNoise* noise, const char* filename,
        int* status);
static void load_noise_freqs(oskar_SettingsSystemNoiseFreq* freq, QSettings& s,
        int* status);
static void load_noise_rms(oskar_SettingsSystemNoiseRMS* value, QSettings& s,
        int* status);
static void get_filename(char** filename, const QSettings& s,
        const QString& key, const QString& defaultValue = "");


extern "C"
void oskar_settings_load_interferometer(oskar_SettingsInterferometer* settings,
        const char* filename, int* status)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);
    QString temp;

    if (*status) return;

    s.beginGroup("interferometer");
    {
        settings->channel_bandwidth_hz =
                s.value("channel_bandwidth_hz").toDouble();
        settings->time_average_sec =
                s.value("time_average_sec", 0.0).toDouble();
        settings->max_time_samples_per_block =
                s.value("max_time_samples_per_block", 10).toInt();
        temp = s.value("correlation_type", "Cross-correlations").toString().toUpper();
        if (temp.startsWith("C"))
            settings->correlation_type = OSKAR_CORRELATION_TYPE_CROSS;
        else if (temp.startsWith("A"))
            settings->correlation_type = OSKAR_CORRELATION_TYPE_AUTO;
        else if (temp.startsWith("B"))
            settings->correlation_type = OSKAR_CORRELATION_TYPE_BOTH;
        else
            *status = OSKAR_ERR_SETTINGS_INTERFEROMETER;

        // Get UV filter parameters.
        temp = s.value("uv_filter_min", "min").toString().toUpper();
        settings->uv_filter_min = (temp == "MIN" ? 0.0 : temp.toDouble());
        temp = s.value("uv_filter_max", "max").toString().toUpper();
        settings->uv_filter_max = (temp == "MAX" ? -1.0 : temp.toDouble());
        temp = s.value("uv_filter_units", "Wavelengths").toString().toUpper();
        if (temp.startsWith("M"))
            settings->uv_filter_units = OSKAR_METRES;
        else if (temp.startsWith("W"))
            settings->uv_filter_units = OSKAR_WAVELENGTHS;

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

        settings->force_polarised_ms = s.value("force_polarised_ms", false).toBool();
    }
    s.endGroup();

    load_noise(&settings->noise, filename, status);
}


static void load_noise(oskar_SettingsSystemNoise* noise, const char* filename,
        int* status)
{
    QByteArray t;
    QString temp;
    QSettings s(QString(filename), QSettings::IniFormat);

    s.beginGroup("interferometer");
    {
        s.beginGroup("noise");
        {
            /* Enable / disable */
            noise->enable = s.value("enable", false).toBool();

            /* Seed */
            temp = s.value("seed", 1).toString().toUpper();
            noise->seed = (temp == "TIME" || temp.toInt() < 1) ?
                    (int)time(NULL) : temp.toInt();

            /* Load frequency settings */
            load_noise_freqs(&noise->freq, s, status);

            /* Values */
            load_noise_rms(&noise->rms, s, status);
        }
        s.endGroup(); // noise
    }
    s.endGroup(); // interferometer
}

static void load_noise_freqs(oskar_SettingsSystemNoiseFreq* freq,
        QSettings& s, int* status)
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
    {
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
        return;
    }

    s.beginGroup("freq");
    {
        get_filename(&freq->file, s, "file");
        freq->number = s.value("number").toInt();
        freq->start  = s.value("start").toDouble();
        freq->inc    = s.value("inc").toDouble();
    }
    s.endGroup(); // freq
}


static void load_noise_rms(oskar_SettingsSystemNoiseRMS* rms,
        QSettings& s, int* status)
{
    QString temp;

    if (*status) return;

    temp = s.value("rms", "T").toString().toUpper();
    if (temp.startsWith("T"))
        rms->specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    else if (temp.startsWith("D"))
        rms->specification = OSKAR_SYSTEM_NOISE_DATA_FILE;
    else if (temp.startsWith("R"))
        rms->specification = OSKAR_SYSTEM_NOISE_RANGE;
    else
    {
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
        return;
    }

    s.beginGroup("rms");
    {
        get_filename(&rms->file, s, "file");
        rms->start  = s.value("start").toDouble();
        rms->end    = s.value("end").toDouble();
    }
    s.endGroup();
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
