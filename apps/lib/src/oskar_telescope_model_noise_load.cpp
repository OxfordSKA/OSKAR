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


#include "apps/lib/oskar_telescope_model_noise_load.h"

#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_telescope_model_resize.h"
#include "station/oskar_system_noise_model_load.h"
#include "station/oskar_station_model_init.h"

#include "utility/oskar_log_message.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_realloc.h"

#include <QtCore/QtCore>

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cfloat>

#ifdef __cplusplus
extern "C" {
#endif

// System noise filenames.
static const char freq_file[]        = "noise_frequencies.txt";
static const char rms_file[]         = "rms.txt";
static const char sensitivity_file[] = "sensitivity.txt";
static const char t_sys_file[]       = "t_sys.txt";
static const char area_file[]        = "area.txt";
static const char efficiency_file[]  = "efficiency.txt";

// Private function prototypes
static int load_directories(oskar_TelescopeModel* telescope, oskar_Log* log,
        const oskar_Settings* settings, const QDir& cwd,
        oskar_StationModel* station, int depth, QHash<QString, QString>& files,
        QHash<QString, oskar_Mem*>& loaded);
static int load_noise_freqs(const oskar_Settings* s, oskar_Mem* freqs,
        const QString& filepath);
static void update_noise_files(QHash<QString, QString>& files, const QDir& dir);
static int load_noise_rms(const oskar_Settings* s,
        oskar_SystemNoiseModel* noise, QHash<QString, QString>& data_files,
        QHash<QString, oskar_Mem*>& loaded);
static int sensitivity_to_rms(oskar_Mem* rms, const oskar_Mem* sensitivity,
        int num_freqs, double bandwidth, double integration_time);
static int t_sys_to_rms(oskar_Mem* rms, const oskar_Mem* t_sys,
        const oskar_Mem* area, const oskar_Mem* efficiency, int num_freqs,
        double bandwidth, double integration_time);

static int evaluate_range(oskar_Mem* values, int num_values, double start, double end);

int oskar_telescope_model_noise_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_Settings* settings)
{
    if (!settings->interferometer.noise.enable)
        return OSKAR_SUCCESS;

    QHash<QString, QString> files;
    QHash<QString, oskar_Mem*> loaded;

    QDir telescope_dir(settings->telescope.input_directory);
    if (!telescope_dir.exists()) return OSKAR_ERR_FILE_IO;

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    int status = OSKAR_SUCCESS;

    // Load element data by scanning the directory structure and loading
    // the element files deepest in the tree.
    status = load_directories(telescope, log, settings, telescope_dir, NULL,
            0, files, loaded);
    if (status)
    {
        oskar_log_error(log, "Loading noise files (%s).", oskar_get_error_string(status));
        return status;
    }

    return OSKAR_SUCCESS;
}


// Private functions

static int load_directories(oskar_TelescopeModel* telescope, oskar_Log* log,
        const oskar_Settings* settings, const QDir& cwd,
        oskar_StationModel* station, int depth, QHash<QString, QString>& files,
        QHash<QString, oskar_Mem*>& loaded)
{
    int err = OSKAR_SUCCESS;

    // Don't go below depth 1 (stations) as currently
    // OSKAR has no mechanism to deal with sub-station detector noise.
    if (depth > 1) return OSKAR_SUCCESS;

    // Update the dictionary of noise files for the current station directory.
    update_noise_files(files, cwd);

    // Get a list of the child stations.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_children = children.size();

    // If the station / child arrays haven't been allocated
    // (by oskar_telescope_load_config() for example), allocate them.
    if (depth == 0 && telescope->station == NULL)
    {
        oskar_telescope_model_resize(telescope, num_children, &err);
    }
    else if (depth > 0 && num_children > 0 && station->child == NULL)
    {
        int type = oskar_telescope_model_type(telescope);
        station->child = (oskar_StationModel*) malloc(num_children *
                sizeof(oskar_StationModel));
        if (!station->child)
            return OSKAR_ERR_MEMORY_ALLOC_FAILURE;

        for (int i = 0; i < num_children; ++i)
        {
            oskar_station_model_init(&station->child[i], type,
                    OSKAR_LOCATION_CPU, 0, &err);
        }
    }
    if (err) return err;

    // Load noise frequency values. (noise files can only be at depth 0).
    if (depth == 0)
    {
        // Load the values into the memory of station 0 and copy to
        // noise structures of other stations.
        oskar_Mem* freqs = &(telescope->station[0].noise.frequency);
        err = load_noise_freqs(settings, freqs, files[freq_file]);
        if (err) return err;
        for (int i = 1; i < num_children; ++i)
        {
            oskar_Mem* dst = &(telescope->station[i].noise.frequency);
            oskar_mem_copy(dst, freqs, &err);
            if (err) return err;
        }
    }

    // Noise files can't currently be deeper than depth 1 (stations).
    if (num_children == 0 && depth <= 1)
    {
        int err = load_noise_rms(settings, &station->noise, files, loaded);
        if (err) return err;
    }

    // Loop over, and descend into the child stations.
    for (int i = 0; i < num_children; ++i)
    {
        // Get a pointer to the child station.
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Get the child directory.
        QDir child_dir(cwd.filePath(children[i]));

        // Load this (child) station.
        err = load_directories(telescope, log, settings, child_dir, s,
                depth + 1, files, loaded);
        if (err) return err;
    }


    return OSKAR_SUCCESS;
}

static int load_noise_freqs(const oskar_Settings* s, oskar_Mem* freqs,
        const QString& filepath)
{
    const oskar_SettingsSystemNoise* noise = &s->interferometer.noise;
    const oskar_SettingsObservation* obs = &s->obs;
    int err = OSKAR_SUCCESS;

    // Load frequency data array from a file.
    if (noise->freq.specification == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL ||
            noise->freq.specification == OSKAR_SYSTEM_NOISE_DATA_FILE)
    {
        QByteArray filename;
        if (noise->freq.specification == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL)
            filename = filepath.toLatin1();
        else
            filename = QByteArray(noise->freq.file);

        // Check if the file exists
        if (!QFile::exists(QString(filename)))
            return OSKAR_ERR_FILE_IO;

        // Load the file
        err = oskar_system_noise_model_load(freqs, filename.constData());
        if (err) return err;
    }

    // Generate frequency data array.
    else
    {
        int num_freqs = 0;
        double start = 0.0, inc = 0.0;
        if (noise->freq.specification == OSKAR_SYSTEM_NOISE_OBS_SETTINGS)
        {
            num_freqs = obs->num_channels;
            start = obs->start_frequency_hz;
            inc = obs->frequency_inc_hz;;
        }
        else if (noise->freq.specification == OSKAR_SYSTEM_NOISE_RANGE)
        {
            num_freqs = noise->freq.number;
            start = noise->freq.start;
            inc = noise->freq.inc;
        }
        oskar_mem_realloc(freqs, num_freqs, &err);
        if (err) return err;
        if (freqs->type == OSKAR_DOUBLE)
        {
            double* freqs_ = (double*)freqs->data;
            for (int i = 0; i < num_freqs; ++i)
            {
                freqs_[i] = start + i * inc;
            }
        }
        else
        {
            float* freqs_ = (float*)freqs->data;
            for (int i = 0; i < num_freqs; ++i)
            {
                freqs_[i] = start + i * inc;
            }
        }
    }

    return OSKAR_SUCCESS;
}

static void update_noise_files(QHash<QString, QString>& files, const QDir& dir)
{
    if (dir.exists(freq_file))
        files[QString(freq_file)] = dir.absoluteFilePath(freq_file);

    if (dir.exists(rms_file))
        files[QString(rms_file)] = dir.absoluteFilePath(rms_file);

    if (dir.exists(sensitivity_file))
        files[QString(sensitivity_file)] = dir.absoluteFilePath(sensitivity_file);

    if (dir.exists(t_sys_file))
        files[QString(t_sys_file)] = dir.absoluteFilePath(t_sys_file);

    if (dir.exists(area_file))
        files[QString(area_file)] = dir.absoluteFilePath(area_file);

    if (dir.exists(efficiency_file))
        files[QString(efficiency_file)] = dir.absoluteFilePath(efficiency_file);
}

static int load_noise_rms(const oskar_Settings* settings,
        oskar_SystemNoiseModel* noise, QHash<QString, QString>& data_files,
        QHash<QString, oskar_Mem*>& /*loaded*/)
{
    const oskar_SettingsSystemNoise* ns = &settings->interferometer.noise;
    int type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    int err = OSKAR_SUCCESS;
    oskar_Mem* stddev = &noise->rms;
    QByteArray file;
    int num_freqs = noise->frequency.num_elements;
    oskar_Mem t_sys(type, OSKAR_LOCATION_CPU, num_freqs);
    double integration_time = settings->obs.length_seconds /
            (double)settings->obs.num_time_steps;
    double bandwidth = settings->interferometer.channel_bandwidth_hz;

    if (bandwidth < DBL_MIN || integration_time < DBL_MIN)
    {
        return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
    }

    // Telescope model priority.
    switch (ns->value.specification)
    {
        // Default (telescope model) priority
        case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
        {
            // RMS
            if (QFile::exists(data_files[rms_file]))
            {
                file = data_files[rms_file].toAscii();
                err = oskar_system_noise_model_load(stddev, file.constData());
                if (err) return err;
            }
            // Sensitivity
            else if (QFile::exists(data_files[sensitivity_file]))
            {

                oskar_Mem sensitivity(type, OSKAR_LOCATION_CPU, num_freqs);
                file = data_files[sensitivity_file].toAscii();
                err = oskar_system_noise_model_load(&sensitivity, file.constData());
                if (err) return err;

                err = sensitivity_to_rms(&noise->rms, &sensitivity,
                        num_freqs, bandwidth, integration_time);
                if (err) return err;
            }
            // T_sys, A_eff and efficiency
            else if (QFile::exists(data_files[t_sys_file]) &&
                    QFile::exists(data_files[area_file]) &&
                    QFile::exists(data_files[efficiency_file]))
            {
                file = data_files[t_sys_file].toAscii();
                err = oskar_system_noise_model_load(&t_sys, file.constData());
                if (err) return err;

                oskar_Mem area(type, OSKAR_LOCATION_CPU, num_freqs);
                file = data_files[area_file].toAscii();
                err = oskar_system_noise_model_load(&area, file.constData());
                if (err) return err;

                oskar_Mem efficiency(type, OSKAR_LOCATION_CPU, num_freqs);
                file = data_files[efficiency_file].toAscii();
                err = oskar_system_noise_model_load(&efficiency, file.constData());
                if (err) return err;

                // FIXME update for efficiency....
                err = t_sys_to_rms(&noise->rms, &t_sys, &area, &efficiency,
                        num_freqs, bandwidth, integration_time);
                if (err) return err;
            }
            else
                return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
        }

        // RMS priority
        case OSKAR_SYSTEM_NOISE_RMS:
        {
            switch (ns->value.rms.override)
            {
                case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                {
                    file = data_files[rms_file].toAscii();
                    err = oskar_system_noise_model_load(stddev, file.constData());
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_DATA_FILE:
                {
                    const char* file = ns->value.rms.file;
                    err = oskar_system_noise_model_load(stddev, file);
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_RANGE:
                {
                    double start = ns->value.rms.start;
                    double end = ns->value.rms.end;
                    err = evaluate_range(stddev, num_freqs, start, end);
                    if (err) return err;
                    break;
                }
                default:
                    return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            };
            break;
        }

        // Sensitivity priority
        case OSKAR_SYSTEM_NOISE_SENSITIVITY:
        {
            oskar_Mem sensitivity(type, OSKAR_LOCATION_CPU, num_freqs);
            switch (ns->value.sensitivity.override)
            {
                case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                {
                    file = data_files[sensitivity_file].toAscii();
                    err = oskar_system_noise_model_load(&sensitivity, file.constData());
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_DATA_FILE:
                {
                    file = ns->value.sensitivity.file;
                    err = oskar_system_noise_model_load(&sensitivity, file);
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_RANGE:
                {
                    double start = ns->value.sensitivity.start;
                    double end = ns->value.sensitivity.end;
                    err = evaluate_range(&sensitivity, num_freqs, start, end);
                    if (err) return err;
                    break;
                }
                default:
                    return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            };
            err = sensitivity_to_rms(stddev, &sensitivity, num_freqs,
                    bandwidth, integration_time);
            if (err) return err;
            break;
        }

        // Temperature, area, and efficiency priority
        case OSKAR_SYSTEM_NOISE_SYS_TEMP:
        {
            switch (ns->value.t_sys.override)
            {
                case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                {
                    file = data_files[sensitivity_file].toAscii();
                    err = oskar_system_noise_model_load(&t_sys, file.constData());
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_DATA_FILE:
                {
                    file = ns->value.t_sys.file;
                    err = oskar_system_noise_model_load(&t_sys, file);
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_RANGE:
                {
                    double start = ns->value.t_sys.start;
                    double end = ns->value.t_sys.end;
                    err = evaluate_range(&t_sys, num_freqs, start, end);
                    if (err) return err;
                    break;
                }
                default:
                    return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            }; // switch (t_sys.override)

            oskar_Mem area(type, OSKAR_LOCATION_CPU, num_freqs);
            switch (ns->value.area.override)
            {
                case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                {
                    file = data_files[area_file].toAscii();
                    err = oskar_system_noise_model_load(&area, file.constData());
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_DATA_FILE:
                {
                    file = ns->value.area.file;
                    err = oskar_system_noise_model_load(&area, file);
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_RANGE:
                {
                    double start = ns->value.area.start;
                    double end = ns->value.area.end;
                    err = evaluate_range(&area, num_freqs, start, end);
                    if (err) return err;
                    break;
                }
                default:
                    return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            }; // switch (area.override)

            oskar_Mem efficiency(type, OSKAR_LOCATION_CPU, num_freqs);
            switch (ns->value.efficiency.override)
            {
                case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
                {
                    file = data_files[area_file].toAscii();
                    err = oskar_system_noise_model_load(&efficiency, file.constData());
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_DATA_FILE:
                {
                    file = ns->value.efficiency.file;
                    err = oskar_system_noise_model_load(&efficiency, file);
                    if (err) return err;
                    break;
                }
                case OSKAR_SYSTEM_NOISE_RANGE:
                {
                    double start = ns->value.efficiency.start;
                    double end = ns->value.efficiency.end;
                    err = evaluate_range(&efficiency, num_freqs, start, end);
                    if (err) return err;
                    break;
                }
                default:
                    return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            }; // switch (efficiency.override)

            err = t_sys_to_rms(stddev, &t_sys, &area, &efficiency, num_freqs,
                    bandwidth, integration_time);
            if (err) return err;

            break;
        }
        default:
        {
            return OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
        }
    }; // [ switch(noise.value.specification) ]

    // Sanity check...
    if (num_freqs != stddev->num_elements)
    {
        return OSKAR_ERR_SETUP_FAIL_TELESCOPE;
    }

    return OSKAR_SUCCESS;
}

static int sensitivity_to_rms(oskar_Mem* rms, const oskar_Mem* sensitivity,
        int num_freqs, double bandwidth, double integration_time)
{
    int err = 0; /* FIXME Replace with status code. */

    double factor = 1.0 / sqrt(2.0 * bandwidth * integration_time);

    if (rms == NULL || sensitivity == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (sensitivity->num_elements != num_freqs)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (rms->num_elements != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, &err);
        if (err) return err;
    }

    if (sensitivity->type == OSKAR_DOUBLE && rms->type == OSKAR_DOUBLE)
    {
        const double* sensitivity_ = (const double*)sensitivity->data;
        double* stddev_ = (double*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sensitivity_[i] * factor;
    }
    else if (sensitivity->type == OSKAR_SINGLE && rms->type == OSKAR_SINGLE)
    {
        const float* sensitivity_ = (const float*)sensitivity->data;
        float* stddev_ = (float*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sensitivity_[i] * factor;
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}

static int t_sys_to_rms(oskar_Mem* rms, const oskar_Mem* t_sys,
        const oskar_Mem* area, const oskar_Mem* efficiency, int num_freqs,
        double bandwidth, double integration_time)
{
    double k_B = 1.3806488e-23;
    double factor = (2.0 * k_B * 1.0e26) / sqrt(2.0 * bandwidth * integration_time);
    int err = 0; /* FIXME Replace with status code. */

    if (rms == NULL || t_sys == NULL || area == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (t_sys->num_elements != num_freqs || area->num_elements != num_freqs)
        return OSKAR_ERR_DIMENSION_MISMATCH;

    if (rms->num_elements != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, &err);
        if (err) return err;
    }

    if (t_sys->type == OSKAR_DOUBLE && area->type == OSKAR_DOUBLE && rms->type == OSKAR_DOUBLE)
    {
        const double* t_sys_ = (const double*)t_sys->data;
        const double* area_ = (const double*)area->data;
        const double* efficiency_ = (const double*)efficiency->data;
        double* rms_ = (double*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * efficiency_[i])) * factor;
    }
    else if (t_sys->type == OSKAR_SINGLE && area->type == OSKAR_SINGLE && rms->type == OSKAR_SINGLE)
    {
        const float* t_sys_ = (const float*)t_sys->data;
        const float* area_ = (const float*)area->data;
        const float* efficiency_ = (const float*)efficiency->data;
        float* rms_ = (float*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * efficiency_[i])) * factor;
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}

static int evaluate_range(oskar_Mem* values, int num_values, double start, double end)
{
    int err = 0; /* FIXME Replace with status code. */
    if (!values) return OSKAR_ERR_INVALID_ARGUMENT;

    double inc = (end - start) / (double)num_values;

    if (values->num_elements != num_values)
    {
        oskar_mem_realloc(values, num_values, &err);
        if (err) return err;
    }

    if (values->type == OSKAR_DOUBLE)
    {
        double* values_ = (double*)values->data;
        for (int i = 0; i < num_values; ++i)
        {
            values_[i] = start + i * inc;
        }
    }
    else if (values->type == OSKAR_SINGLE)
    {
        float* values_ = (float*)values->data;
        for (int i = 0; i < num_values; ++i)
        {
            values_[i] = start + i * inc;
        }
    }
    else
        return OSKAR_ERR_BAD_DATA_TYPE;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif

