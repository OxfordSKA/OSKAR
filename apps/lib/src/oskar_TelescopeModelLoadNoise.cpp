/*
 * Copyright (c) 2013, The University of Oxford
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


#include "apps/lib/oskar_TelescopeModelLoadNoise.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_copy.h"
#include "utility/oskar_mem_realloc.h"
#include "interferometry/oskar_telescope_model_type.h"

#include <QtCore/QDir>
#include <QtCore/QFile>

#include <cfloat>
#include <cmath>

oskar_TelescopeModelLoadNoise::oskar_TelescopeModelLoadNoise(
        const oskar_Settings* settings)
: oskar_AbstractTelescopeFileLoader()
{
    files_[FREQ] = "noise_frequencies.txt";
    files_[RMS]  = "rms.txt";
    files_[SENSITIVITY] = "sensitivity.txt";
    files_[TSYS] = "t_sys.txt";
    files_[AREA] = "area.txt";
    files_[EFFICIENCY] = "efficiency.txt";
    settings_ = settings;
}

oskar_TelescopeModelLoadNoise::~oskar_TelescopeModelLoadNoise()
{
    int status = OSKAR_SUCCESS;
    oskar_mem_free(&freqs_, &status);
    // Note: status code ignored! ...
}


// Depth = 0
// - Set up frequency data as this is the same for all stations
//   and if defined by files these have to be at depth 0.
void oskar_TelescopeModelLoadNoise::load(oskar_TelescopeModel* telescope,
        const QDir& cwd, int num_subdirs, QHash<QString, QString>& filemap,
        int* status)
{
    if (*status || !settings_->interferometer.noise.enable)
        return;

    dataType_ = oskar_telescope_model_type(telescope);

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);

    // Set up noise frequency values (this only happens at depth = 0)
    oskar_mem_init(&freqs_, dataType_, 0, 0, 1, status);
    getNoiseFreqs_(&freqs_, filemap[files_[FREQ]], status);

    // No sub-directories (the other load function is never called)
    if (num_subdirs == 0)
    {
        for (int i = 0; i < telescope->num_stations; ++i)
        {
            oskar_StationModel* station = &telescope->station[i];
            oskar_mem_copy(&station->noise.frequency, &freqs_, status);
            setNoiseRMS_(&station->noise, filemap, status);
        }

    }
}



// Depth > 0
void oskar_TelescopeModelLoadNoise::load(oskar_StationModel* station,
        const QDir& cwd, int /*num_subdirs*/, int depth,
        QHash<QString, QString>& filemap, int* status)
{
    if (*status || !settings_->interferometer.noise.enable)
        return;

    if (*status) return;

    // Ignore noise files defined deeper than at the station level (depth == 1)
    // - Currently, noise is implemented as a additive term per station
    //   into the visibilities so using files at any other depth would be
    //   meaningless.
    if (depth > 1)
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);


    // Set the frequency noise data field of the station structure.
    oskar_mem_copy(&station->noise.frequency, &freqs_, status);


    // Set the noise RMS based on files or settings.
    setNoiseRMS_(&station->noise, filemap, status);
}


// -- private functions -------------------------------------------------------

void oskar_TelescopeModelLoadNoise::updateFileMap_(
        QHash<QString, QString>& filemap, const QDir& cwd)
{
    foreach(QString file, files_.values()) {
        filemap[file] = cwd.absoluteFilePath(file);
    }
}

void oskar_TelescopeModelLoadNoise::getNoiseFreqs_(oskar_Mem* freqs,
        const QString& filepath, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoise& noise = settings_->interferometer.noise;
    const oskar_SettingsObservation& obs = settings_->obs;

    int freq_spec = noise.freq.specification;

    // Case 1) Load frequency data array form a file
    if (freq_spec == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL ||
            freq_spec == OSKAR_SYSTEM_NOISE_DATA_FILE)
    {
        // Get the filename to load
        QByteArray filename;
        if (noise.freq.specification == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL)
            filename = filepath.toLatin1();
        else
            filename = QByteArray(noise.freq.file);

        // Check the file exists.
        if (!QFile::exists(QString(filename))) {
            *status = OSKAR_ERR_FILE_IO;
            return;
        }

        // Load the file
        oskar_system_noise_model_load(freqs, filename.constData(), status);
    }

    // Case 2) generate the frequency data.
    else
    {
        int num_freqs = 0;
        double start = 0, inc = 0;
        if (freq_spec == OSKAR_SYSTEM_NOISE_OBS_SETTINGS)
        {
            num_freqs = obs.num_channels;
            start = obs.start_frequency_hz;
            inc = obs.frequency_inc_hz;
        }
        else if (freq_spec == OSKAR_SYSTEM_NOISE_RANGE)
        {
            num_freqs = noise.freq.number;
            start = noise.freq.start;
            inc = noise.freq.inc;
        }
        if (num_freqs == 0) {
            *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
            return;
        }
        oskar_mem_realloc(freqs, num_freqs, status);
        if (*status) return;
        if (freqs->type == OSKAR_DOUBLE)
        {
            for (int i = 0; i < num_freqs; ++i)
                ((double*)freqs->data)[i] = start + i * inc;
        }
        else
        {
            for (int i = 0; i < num_freqs; ++i)
                ((float*)freqs->data)[i] = start + i * inc;
        }

    }
}

void oskar_TelescopeModelLoadNoise::setNoiseRMS_(
        oskar_SystemNoiseModel* noise_model, const QHash<QString,
        QString>& filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoise& settings_noise = settings_->interferometer.noise;
    oskar_Mem* noise_rms = &noise_model->rms;
    int num_freqs = noise_model->frequency.num_elements;
    // Note: the previous noise loader implementation had integration time as
    // obs_length / number of snapshots which was wrong!
    double integration_time = settings_->interferometer.time_average_sec;
    double bandwidth = settings_->interferometer.channel_bandwidth_hz;

    if (bandwidth < DBL_MIN || integration_time < DBL_MIN)
    {
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
        return;
    }

    switch (settings_noise.value.specification)
    {
        case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
            noiseSpecTelescopeModel_(noise_rms, num_freqs, bandwidth,
                    integration_time, filemap, status);
            break;
        case OSKAR_SYSTEM_NOISE_RMS:
            noiseSpecRMS_(noise_rms, num_freqs, filemap, status);
            break;
        case OSKAR_SYSTEM_NOISE_SENSITIVITY:
            noiseSpecSensitivity_(noise_rms, num_freqs, bandwidth,
                    integration_time, filemap, status);
            break;
        case OSKAR_SYSTEM_NOISE_SYS_TEMP:
            noiseSpecTsys_(noise_rms, num_freqs, bandwidth, integration_time,
                    filemap, status);
            break;
        default:
            *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
            return;
    };
}


// Load noise files from the telescope model using default noise spec. priority.
void oskar_TelescopeModelLoadNoise::noiseSpecTelescopeModel_(oskar_Mem* noise_rms,
        int num_freqs, double bandwidth_hz, double integration_time_sec,
        const QHash<QString, QString>& filemap, int* status)
{
    if (*status) return;

    QByteArray filename;
    int loc = OSKAR_LOCATION_CPU;

    // RMS
    if (QFile::exists(filemap[files_[RMS]]))
    {
        filename = filemap[files_[RMS]].toAscii();
        oskar_system_noise_model_load(noise_rms, filename.constData(), status);
    }

    // Sensitivity
    else if (QFile::exists(filemap[files_[SENSITIVITY]]))
    {
        oskar_Mem sens;
        oskar_mem_init(&sens, dataType_, loc, num_freqs, OSKAR_TRUE, status);
        filename = filemap[files_[SENSITIVITY]].toAscii();
        const char* file = filename.constData();
        oskar_system_noise_model_load(&sens, file, status);
        sensitivity_to_rms_(noise_rms, &sens, num_freqs, bandwidth_hz,
                integration_time_sec, status);
        oskar_mem_free(&sens, status);
    }

    // T_sys, A_eff and efficiency
    else if (QFile::exists(filemap[files_[TSYS]]) &&
            QFile::exists(filemap[files_[AREA]]) &&
            QFile::exists(filemap[files_[EFFICIENCY]]))
    {
        oskar_Mem t_sys, area, efficiency;
        oskar_mem_init(&t_sys, dataType_, loc, num_freqs, 0, status);
        oskar_mem_init(&area, dataType_, loc, num_freqs, 0, status);
        oskar_mem_init(&efficiency, dataType_, loc, num_freqs, 0, status);
        filename = filemap[files_[TSYS]].toAscii();
        oskar_system_noise_model_load(&t_sys, filename.constData(), status);
        filename = filemap[files_[AREA]].toAscii();
        oskar_system_noise_model_load(&area, filename.constData(), status);
        filename = filemap[files_[EFFICIENCY]].toAscii();
        oskar_system_noise_model_load(&efficiency, filename.constData(),
                status);
        t_sys_to_rms_(noise_rms, &t_sys, &area, &efficiency,
                num_freqs, bandwidth_hz, integration_time_sec, status);
        oskar_mem_free(&t_sys, status);
        oskar_mem_free(&area, status);
        oskar_mem_free(&efficiency, status);
    }
    else
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
    }
}

void oskar_TelescopeModelLoadNoise::noiseSpecRMS_(oskar_Mem* rms, int num_freqs,
        const QHash<QString, QString>& filemap, int* status)
{
    if (*status) return;

    QByteArray filename;
    const oskar_SettingsSystemNoiseType& settingsRMS =
            settings_->interferometer.noise.value.rms;

    switch (settingsRMS.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
            filename = filemap[files_[RMS]].toAscii();
            oskar_system_noise_model_load(rms, filename.constData(), status);
            break;
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_system_noise_model_load(rms, settingsRMS.file, status);
            break;
        case OSKAR_SYSTEM_NOISE_RANGE:
            evaluate_range_(rms, num_freqs, settingsRMS.start, settingsRMS.end,
                    status);
            break;
        default:
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
    }
}


void oskar_TelescopeModelLoadNoise::noiseSpecSensitivity_(oskar_Mem* rms,
        int num_freqs, double bandwidth_hz, double integration_time_sec,
        QHash<QString, QString> filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoiseType& s =
            settings_->interferometer.noise.value.sensitivity;
    oskar_Mem sens;
    oskar_mem_init(&sens, dataType_, OSKAR_LOCATION_CPU, num_freqs, 1, status);
    switch (s.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
        {
            QByteArray filename = filemap[files_[SENSITIVITY]].toAscii();
            oskar_system_noise_model_load(&sens, filename.constData(), status);
            break;
        }
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_system_noise_model_load(&sens, s.file, status);
            break;
        case OSKAR_SYSTEM_NOISE_RANGE:
            evaluate_range_(&sens, num_freqs, s.start, s.end, status);
            break;
        default:
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
    };
    sensitivity_to_rms_(rms, &sens, num_freqs, bandwidth_hz,
            integration_time_sec, status);
    oskar_mem_free(&sens, status);
}

void oskar_TelescopeModelLoadNoise::noiseSpecTsys_(oskar_Mem* rms, int num_freqs,
        double bandwidth_hz, double integration_time_sec,
        QHash<QString, QString> filemap, int* status)
{
    if (*status) return;

    oskar_Mem t_sys, area, efficiency;
    int loc = OSKAR_LOCATION_CPU;
    oskar_mem_init(&t_sys, dataType_, loc, num_freqs, 1, status);
    oskar_mem_init(&area, dataType_, loc, num_freqs, 1, status);
    oskar_mem_init(&efficiency, dataType_, loc, num_freqs, 1, status);

    const oskar_SettingsSystemNoiseValue& s = settings_->interferometer.noise.value;

    // Load/evaluate T_sys values.

    switch (s.t_sys.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
        {
            QByteArray filename = filemap[files_[TSYS]].toAscii();
            oskar_system_noise_model_load(&t_sys, filename.constData(), status);
            break;
        }
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_system_noise_model_load(&t_sys, s.t_sys.file, status);
            break;
        case OSKAR_SYSTEM_NOISE_RANGE:
            evaluate_range_(&t_sys, num_freqs, s.t_sys.start, s.t_sys.end, status);
            break;
        default:
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
    }

    // Load/evaluate effective area values.
    switch (s.area.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
        {
            QByteArray filename = filemap[files_[AREA]].toAscii();
            oskar_system_noise_model_load(&area, filename.constData(), status);
            break;
        }
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_system_noise_model_load(&area, s.area.file, status);
            break;
        case OSKAR_SYSTEM_NOISE_RANGE:
            evaluate_range_(&area, num_freqs, s.area.start, s.area.end, status);
            break;
        default:
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
    }

    switch (s.efficiency.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
        {
            QByteArray filename = filemap[files_[AREA]].toAscii();
            oskar_system_noise_model_load(&efficiency, filename.constData(),
                    status);
            break;
        }
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_system_noise_model_load(&efficiency, s.efficiency.file, status);
            break;
        case OSKAR_SYSTEM_NOISE_RANGE:
            evaluate_range_(&efficiency, num_freqs, s.efficiency.start,
                    s.efficiency.end, status);
            break;
        default:
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
            break;
    };

    t_sys_to_rms_(rms, &t_sys, &area, &efficiency, num_freqs, bandwidth_hz,
            integration_time_sec, status);

    oskar_mem_free(&t_sys, status);
    oskar_mem_free(&area, status);
    oskar_mem_free(&efficiency, status);
}

void oskar_TelescopeModelLoadNoise::sensitivity_to_rms_(oskar_Mem* rms,
        const oskar_Mem* sensitivity, int num_freqs, double bandwidth_hz,
        double integration_time_sec, int* status)
{
    if (*status) return;

    if (rms->num_elements != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, status);
        if (*status) return;
    }

    double factor = 1.0 / std::sqrt(2.0 * bandwidth_hz * integration_time_sec);
    if (dataType_ == OSKAR_DOUBLE)
    {
        const double* sensitivity_ = (const double*)sensitivity->data;
        double* stddev_ = (double*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sensitivity_[i] * factor;
    }
    else if (dataType_ == OSKAR_SINGLE)
    {
        const float* sensitivity_ = (const float*)sensitivity->data;
        float* stddev_ = (float*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sensitivity_[i] * factor;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}


void oskar_TelescopeModelLoadNoise::t_sys_to_rms_(oskar_Mem* rms,
        const oskar_Mem* t_sys, const oskar_Mem* area,
        const oskar_Mem* efficiency, int num_freqs, double bandwidth,
        double integration_time, int* status)
{
    if (*status) return;

    double k_B = 1.3806488e-23;

    /* Get type and check consistency. */
    int type = rms->type;
    if (t_sys->type != type || area->type != type || efficiency->type != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if (t_sys->num_elements != num_freqs || area->num_elements != num_freqs)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    if (rms->num_elements != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, status);
        if (*status) return;
    }

    double factor = (2.0 * k_B * 1.0e26) / sqrt(2.0 * bandwidth * integration_time);
    if (type == OSKAR_DOUBLE)
    {
        const double* t_sys_ = (const double*)t_sys->data;
        const double* area_ = (const double*)area->data;
        const double* efficiency_ = (const double*)efficiency->data;
        double* rms_ = (double*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * efficiency_[i])) * factor;
    }
    else if (type == OSKAR_SINGLE)
    {
        const float* t_sys_ = (const float*)t_sys->data;
        const float* area_ = (const float*)area->data;
        const float* efficiency_ = (const float*)efficiency->data;
        float* rms_ = (float*)rms->data;
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * efficiency_[i])) * factor;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}


void oskar_TelescopeModelLoadNoise::evaluate_range_(oskar_Mem* values,
        int num_values, double start, double end, int* status)
{
    /* Check all inputs. */
    if (!values || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    double inc = (end - start) / (double)num_values;
    if (values->num_elements != num_values)
    {
        oskar_mem_realloc(values, num_values, status);
        if (*status) return;
    }

    if (values->type == OSKAR_DOUBLE)
    {
        double* values_ = (double*)values->data;
        for (int i = 0; i < num_values; ++i)
            values_[i] = start + i * inc;
    }
    else if (values->type == OSKAR_SINGLE)
    {
        float* values_ = (float*)values->data;
        for (int i = 0; i < num_values; ++i)
            values_[i] = start + i * inc;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}


