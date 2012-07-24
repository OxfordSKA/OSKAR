/*
 * Copyright (c) 2012, The University of Oxford
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

#include "apps/lib/oskar_settings_load_interferomter.h"
#include "sky/oskar_date_time_to_mjd.h"
#include "utility/oskar_log_error.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <QtCore/QSettings>
#include <QtCore/QByteArray>
#include <QtCore/QDateTime>
#include <QtCore/QVariant>
#include <QtCore/QDate>
#include <QtCore/QTime>
#include <QtCore/QString>

// =============================================================================
static int load_noise(oskar_SettingsSystemNoise* noise, const char* filename);
static int load_noise_freqs(oskar_SettingsSystemNoiseFreq* freq, QSettings& s);
static int load_noise_values(oskar_SettingsSystemNoiseValue* value, QSettings& s);
static int load_noise_type(oskar_SettingsSystemNoiseType* stddev, QSettings& s,
        const QString& name);
static int load_noise_t_ant(oskar_SettingsSystemNoiseTAnt* t_ant, QSettings& s);
static int load_noise_area(oskar_SettingsSystemNoiseArea* area, QSettings& s);
static void get_filename(char** filename, const QSettings& s, const QString& key,
        const QString& defaultValue = "");
//=============================================================================

extern "C"
int oskar_settings_load_interferometer(oskar_SettingsInterferometer* settings,
        const char* filename)
{
    QByteArray t;
    QSettings s(QString(filename), QSettings::IniFormat);

    s.beginGroup("interferometer");
    {
        settings->channel_bandwidth_hz = s.value("channel_bandwidth_hz").toDouble();
        settings->num_vis_ave     = s.value("num_vis_ave", 1).toInt();
        settings->num_fringe_ave  = s.value("num_fringe_ave", 1).toInt();

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
        s.beginGroup("system_noise");
        {
            /* Enable / disable */
            noise->enable = s.value("enable", false).toBool();

            /* Seed */
            temp = s.value("seed").toString().toUpper();
            noise->seed = (temp == "TIME" || temp.toInt() < 0) ?
                    (int)time(NULL) : temp.toInt();

            /* Area projection */
            noise->area_projection = s.value("area_projection", true).toBool();

            /* Load frequency settings */
            error = load_noise_freqs(&noise->freq, s);
            if (error) return error;

            /* Values */
            error = load_noise_values(&noise->value, s);
            if (error) return error;
        }
        s.endGroup(); // system_noise
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
    else if (temp.startsWith("U"))
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
        s.beginGroup("range");
        {
            freq->number = s.value("number").toInt();
            freq->start  = s.value("start").toDouble();
            freq->inc    = s.value("inc").toDouble();
        }
        s.endGroup(); // range
    }
    s.endGroup(); // freq

    return OSKAR_SUCCESS;
}


static int load_noise_values(oskar_SettingsSystemNoiseValue* value,
        QSettings& s)
{
    int error = OSKAR_SUCCESS;

    QString temp = s.value("spec", "T").toString().toUpper();
    if (temp.startsWith("T"))
        value->specification = OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL;
    else if (temp.startsWith("F"))
        value->specification = OSKAR_SYSTEM_NOISE_STDDEV;
    else if (temp.startsWith("SEN"))
        value->specification = OSKAR_SYSTEM_NOISE_SENSITIVITY;
    else if(temp.startsWith("SYS"))
        value->specification = OSKAR_SYSTEM_NOISE_SYS_TEMP;
    else if(temp.startsWith("C"))
        value->specification = OSKAR_SYSTEM_NOISE_TEMPS;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup("spec");
    {
        error = load_noise_type(&value->stddev, s, "stddev");
        if (error) return error;

        error = load_noise_type(&value->sensitivity, s, "sensitivity");
        if (error) return error;

        error = load_noise_type(&value->t_sys, s, "t_sys");
        if (error) return error;

        s.beginGroup("temp");
        {
            value->t_amb = s.value("ambient", 0.0).toDouble();

            error = load_noise_type(&value->t_rec, s, "receiver");
            if (error) return error;

            error = load_noise_t_ant(&value->t_ant, s);
            if (error) return error;

            error = load_noise_type(&value->radiation_efficiency, s, "radiation_efficiency");
            if (error) return error;
        }
        s.endGroup();

        error = load_noise_area(&value->area, s);
        if (error) return error;
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
        s.beginGroup("range");
        {
            type->start  = s.value("start").toDouble();
            type->end    = s.value("end").toDouble();
        }
        s.endGroup(); // range
    }
    s.endGroup();

    return OSKAR_SUCCESS;
}


static int load_noise_t_ant(oskar_SettingsSystemNoiseTAnt* t_ant, QSettings& s)
{
    QString temp = s.value("antenna", "N").toString().toUpper();
    if (temp.startsWith("N"))
        t_ant->override = OSKAR_SYSTEM_NOISE_NO_OVERRIDE;
    else if (temp.startsWith("D"))
        t_ant->override = OSKAR_SYSTEM_NOISE_DATA_FILE;
    else if (temp.startsWith("R"))
        t_ant->override = OSKAR_SYSTEM_NOISE_RANGE;
    else if (temp.startsWith("S"))
        t_ant->override = OSKAR_SYSTEM_NOISE_SPIX_MODEL;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup("antenna");
    {
        get_filename(&t_ant->file, s, "file");
        s.beginGroup("range");
        {
            t_ant->start  = s.value("start", 0.0).toDouble();
            t_ant->end    = s.value("end", 0.0).toDouble();
        }
        s.endGroup();
        s.beginGroup("model");
        {
            t_ant->model_spix  = s.value("specral_index", 2.75).toDouble();
            t_ant->model_t_408 = s.value("temp_408", 20.0).toDouble();
        }
        s.endGroup();
    }
    s.endGroup();

    return OSKAR_SUCCESS;
}


static int load_noise_area(oskar_SettingsSystemNoiseArea* area, QSettings& s)
{
    QString temp = s.value("area", "N").toString().toUpper();
    if (temp.startsWith("N"))
        area->override = OSKAR_SYSTEM_NOISE_NO_OVERRIDE;
    else if (temp.startsWith("D"))
        area->override = OSKAR_SYSTEM_NOISE_DATA_FILE;
    else if (temp.startsWith("R"))
        area->override = OSKAR_SYSTEM_NOISE_RANGE;
//    else if (temp.startsWith("A"))
//        area->override = OSKAR_SYSTEM_NOISE_AREA_MODEL;
    else
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    s.beginGroup("area");
    {
        get_filename(&area->file, s, "file");
        s.beginGroup("range");
        {
            area->start  = s.value("start", 0.0).toDouble();
            area->end    = s.value("end", 0.0).toDouble();
        }
        s.endGroup();
//        QString temp = s.value("model", "S").toString().toUpper();
//        if (temp.startsWith("S"))
//            area->model_type = OSKAR_SYSTEM_NOISE_AREA_SPARSE;
//        else if (temp.startsWith("D"))
//            area->model_type = OSKAR_SYSTEM_NOISE_AREA_DENSE;
//        else
//            return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
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
