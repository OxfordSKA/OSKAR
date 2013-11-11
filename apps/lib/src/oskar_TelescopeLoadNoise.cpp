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

#include "apps/lib/oskar_TelescopeLoadNoise.h"
#include "apps/lib/oskar_Dir.h"

#include <oskar_Settings.h>
#include <oskar_system_noise_model_load.h>
#include <oskar_file_exists.h>

#include <cfloat>
#include <cassert>
#include <cmath>

using std::map;
using std::string;

oskar_TelescopeLoadNoise::oskar_TelescopeLoadNoise(
        const oskar_Settings* settings)
: oskar_TelescopeLoadAbstract(), dataType_(0)
{
    files_[FREQ] = "noise_frequencies.txt";
    files_[RMS]  = "rms.txt";
    files_[SENSITIVITY] = "sensitivity.txt";
    files_[TSYS] = "t_sys.txt";
    files_[AREA] = "area.txt";
    files_[EFFICIENCY] = "efficiency.txt";
    settings_ = settings;
}

oskar_TelescopeLoadNoise::~oskar_TelescopeLoadNoise()
{
    int status = OSKAR_SUCCESS;
    oskar_mem_free(&freqs_, &status);
    assert(status == OSKAR_SUCCESS);
    // Note: status code ignored in release mode when NDEBUG is defined.
}


// Depth = 0
// - Set up frequency data as this is the same for all stations
//   and if defined by files these have to be at depth 0.
void oskar_TelescopeLoadNoise::load(oskar_Telescope* telescope,
        const oskar_Dir& cwd, int num_subdirs, map<string, string>& filemap,
        int* status)
{
    if (*status || !settings_->interferometer.noise.enable)
        return;

    dataType_ = oskar_telescope_type(telescope);

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);

    // Set up noise frequency values (this only happens at depth = 0)
    oskar_mem_init(&freqs_, dataType_, 0, 0, 1, status);
    getNoiseFreqs_(&freqs_, filemap[files_[FREQ]], status);

    // If no sub-directories (the station load function is never called)
    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_stations(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* s = oskar_telescope_station(telescope, i);
            oskar_SystemNoiseModel* noise = oskar_station_system_noise_model(s);
            oskar_mem_copy(&noise->frequency, &freqs_, status);
            setNoiseRMS_(noise, filemap, status);
        }
    }
}


// Depth > 0
void oskar_TelescopeLoadNoise::load(oskar_Station* station,
        const oskar_Dir& cwd, int /*num_subdirs*/, int depth,
        map<string, string>& filemap, int* status)
{
    if (*status || !settings_->interferometer.noise.enable)
        return;

    // Ignore noise files defined deeper than at the station level (depth == 1)
    // - Currently, noise is implemented as a additive term per station
    //   into the visibilities so using files at any other depth would be
    //   meaningless.
    if (depth > 1)
    {
        *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
        return;
    }

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);

    // Set the frequency noise data field of the station structure.
    oskar_SystemNoiseModel* noise = oskar_station_system_noise_model(station);
    oskar_mem_copy(&noise->frequency, &freqs_, status);

    // Set the noise RMS based on files or settings.
    setNoiseRMS_(noise, filemap, status);
}

string oskar_TelescopeLoadNoise::name() const
{
    return string("noise loader");
}

// -- private functions -------------------------------------------------------

void oskar_TelescopeLoadNoise::updateFileMap_(map<string, string>& filemap,
        const oskar_Dir& cwd)
{
    for (map<FileIds_, string>::const_iterator it = files_.begin();
            it != files_.end(); ++it)
    {
        string file = it->second;
        if (cwd.exists(file))
            filemap[file] = cwd.absoluteFilePath(file);
    }
}

void oskar_TelescopeLoadNoise::getNoiseFreqs_(oskar_Mem* freqs,
        const string& filepath, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoise& noise = settings_->interferometer.noise;
    const oskar_SettingsObservation& obs = settings_->obs;

    int freq_spec = noise.freq.specification;

    // Case 1: Load frequency data array from a file.
    if (freq_spec == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL ||
            freq_spec == OSKAR_SYSTEM_NOISE_DATA_FILE)
    {
        // Get the filename to load.
        string filename;
        if (noise.freq.specification == OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL)
            filename = filepath;
        else
            filename = string(noise.freq.file);

        // Check the file exists.
        if (!oskar_file_exists(filename.c_str()))
        {
            *status = OSKAR_ERR_FILE_IO;
            return;
        }

        // Load the file.
        oskar_system_noise_model_load(freqs, filename.c_str(), status);
    }

    // Case 2: Generate the frequency data.
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
        if (num_freqs == 0)
        {
            *status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE;
            return;
        }
        oskar_mem_realloc(freqs, num_freqs, status);
        if (*status) return;
        if (oskar_mem_type(freqs) == OSKAR_DOUBLE)
        {
            double* f = oskar_mem_double(freqs, status);
            for (int i = 0; i < num_freqs; ++i)
                f[i] = start + i * inc;
        }
        else
        {
            float* f = oskar_mem_float(freqs, status);
            for (int i = 0; i < num_freqs; ++i)
                f[i] = start + i * inc;
        }
    }
}

void oskar_TelescopeLoadNoise::setNoiseRMS_(oskar_SystemNoiseModel* noise_model,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoise& settings_noise = settings_->interferometer.noise;
    oskar_Mem* noise_rms = &noise_model->rms;
    int num_freqs = (int)oskar_mem_length(&noise_model->frequency);
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
void oskar_TelescopeLoadNoise::noiseSpecTelescopeModel_(oskar_Mem* noise_rms,
        int num_freqs, double bandwidth_hz, double integration_time_sec,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    string f_rms, f_sensitivity, f_tsys, f_area, f_efficiency;
    int loc = OSKAR_LOCATION_CPU;

    // Get the filenames out of the map, if they are set.
    if (filemap.count(files_[RMS]))
        f_rms = filemap.at(files_[RMS]);
    if (filemap.count(files_[SENSITIVITY]))
        f_sensitivity = filemap.at(files_[SENSITIVITY]);
    if (filemap.count(files_[TSYS]))
        f_tsys = filemap.at(files_[TSYS]);
    if (filemap.count(files_[AREA]))
        f_area = filemap.at(files_[AREA]);
    if (filemap.count(files_[EFFICIENCY]))
        f_efficiency = filemap.at(files_[EFFICIENCY]);

    // RMS
    if (oskar_file_exists(f_rms.c_str()))
    {
        oskar_system_noise_model_load(noise_rms, f_rms.c_str(), status);
    }

    // Sensitivity
    else if (oskar_file_exists(f_sensitivity.c_str()))
    {
        oskar_Mem sens;
        oskar_mem_init(&sens, dataType_, loc, num_freqs, OSKAR_TRUE, status);
        oskar_system_noise_model_load(&sens, f_sensitivity.c_str(), status);
        sensitivity_to_rms_(noise_rms, &sens, num_freqs, bandwidth_hz,
                integration_time_sec, status);
        oskar_mem_free(&sens, status);
    }

    // T_sys, A_eff and efficiency
    else if (oskar_file_exists(f_tsys.c_str()) &&
            oskar_file_exists(f_area.c_str()) &&
            oskar_file_exists(f_efficiency.c_str()))
    {
        oskar_Mem t_sys, area, efficiency;
        oskar_mem_init(&t_sys, dataType_, loc, num_freqs, 1, status);
        oskar_mem_init(&area, dataType_, loc, num_freqs, 1, status);
        oskar_mem_init(&efficiency, dataType_, loc, num_freqs, 1, status);
        oskar_system_noise_model_load(&t_sys, f_tsys.c_str(), status);
        oskar_system_noise_model_load(&area, f_area.c_str(), status);
        oskar_system_noise_model_load(&efficiency, f_efficiency.c_str(),
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

void oskar_TelescopeLoadNoise::noiseSpecRMS_(oskar_Mem* rms, int num_freqs,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoiseType& settingsRMS =
            settings_->interferometer.noise.value.rms;

    string filename;
    if (filemap.count(files_[RMS]))
        filename = filemap.at(files_[RMS]);

    switch (settingsRMS.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
            oskar_system_noise_model_load(rms, filename.c_str(), status);
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


void oskar_TelescopeLoadNoise::noiseSpecSensitivity_(oskar_Mem* rms,
        int num_freqs, double bandwidth_hz, double integration_time_sec,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoiseType& s =
            settings_->interferometer.noise.value.sensitivity;

    string filename;
    if (filemap.count(files_[SENSITIVITY]))
        filename = filemap.at(files_[SENSITIVITY]);

    oskar_Mem sens;
    oskar_mem_init(&sens, dataType_, OSKAR_LOCATION_CPU, num_freqs, 1, status);
    switch (s.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
            oskar_system_noise_model_load(&sens, filename.c_str(), status);
            break;
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

void oskar_TelescopeLoadNoise::noiseSpecTsys_(oskar_Mem* rms, int num_freqs,
        double bandwidth_hz, double integration_time_sec,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    oskar_Mem t_sys, area, efficiency;
    int loc = OSKAR_LOCATION_CPU;
    oskar_mem_init(&t_sys, dataType_, loc, num_freqs, 1, status);
    oskar_mem_init(&area, dataType_, loc, num_freqs, 1, status);
    oskar_mem_init(&efficiency, dataType_, loc, num_freqs, 1, status);

    const oskar_SettingsSystemNoiseValue& s = settings_->interferometer.noise.value;

    string f_tsys, f_area, f_efficiency;
    if (filemap.count(files_[TSYS]))
        f_tsys = filemap.at(files_[TSYS]);
    if (filemap.count(files_[AREA]))
        f_area = filemap.at(files_[AREA]);
    if (filemap.count(files_[EFFICIENCY]))
        f_efficiency = filemap.at(files_[EFFICIENCY]);

    // Load/evaluate T_sys values.
    switch (s.t_sys.override)
    {
        case OSKAR_SYSTEM_NOISE_NO_OVERRIDE:
            oskar_system_noise_model_load(&t_sys, f_tsys.c_str(), status);
            break;
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
            oskar_system_noise_model_load(&area, f_area.c_str(), status);
            break;
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
            oskar_system_noise_model_load(&efficiency, f_efficiency.c_str(),
                    status);
            break;
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

void oskar_TelescopeLoadNoise::sensitivity_to_rms_(oskar_Mem* rms,
        const oskar_Mem* sensitivity, int num_freqs, double bandwidth_hz,
        double integration_time_sec, int* status)
{
    if (*status) return;

    if ((int)oskar_mem_length(rms) != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, status);
        if (*status) return;
    }

    double factor = 1.0 / std::sqrt(2.0 * bandwidth_hz * integration_time_sec);
    if (dataType_ == OSKAR_DOUBLE)
    {
        const double* sens_ = oskar_mem_double_const(sensitivity, status);
        double* stddev_ = oskar_mem_double(rms, status);
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sens_[i] * factor;
    }
    else if (dataType_ == OSKAR_SINGLE)
    {
        const float* sens_ = oskar_mem_float_const(sensitivity, status);
        float* stddev_ = oskar_mem_float(rms, status);
        for (int i = 0; i < num_freqs; ++i)
            stddev_[i] = sens_[i] * factor;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}


void oskar_TelescopeLoadNoise::t_sys_to_rms_(oskar_Mem* rms,
        const oskar_Mem* t_sys, const oskar_Mem* area,
        const oskar_Mem* efficiency, int num_freqs, double bandwidth,
        double integration_time, int* status)
{
    if (*status) return;

    const double k_B = 1.3806488e-23;

    /* Get type and check consistency. */
    int type = oskar_mem_type(rms);
    if (oskar_mem_type(t_sys) != type || oskar_mem_type(area) != type ||
            oskar_mem_type(efficiency) != type)
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
        return;
    }
    if ((int)oskar_mem_length(t_sys) != num_freqs ||
            (int)oskar_mem_length(area) != num_freqs)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    if ((int)oskar_mem_length(rms) != num_freqs)
    {
        oskar_mem_realloc(rms, num_freqs, status);
        if (*status) return;
    }

    double factor = (2.0 * k_B * 1.0e26) /
            sqrt(2.0 * bandwidth * integration_time);
    if (type == OSKAR_DOUBLE)
    {
        const double* t_sys_ = oskar_mem_double_const(t_sys, status);
        const double* area_ = oskar_mem_double_const(area, status);
        const double* eff_ = oskar_mem_double_const(efficiency, status);
        double* rms_ = oskar_mem_double(rms, status);
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * eff_[i])) * factor;
    }
    else if (type == OSKAR_SINGLE)
    {
        const float* t_sys_ = oskar_mem_float_const(t_sys, status);
        const float* area_ = oskar_mem_float_const(area, status);
        const float* eff_ = oskar_mem_float_const(efficiency, status);
        float* rms_ = oskar_mem_float(rms, status);
        for (int i = 0; i < num_freqs; ++i)
            rms_[i] = (t_sys_[i] / (area_[i] * eff_[i])) * factor;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}


void oskar_TelescopeLoadNoise::evaluate_range_(oskar_Mem* values,
        int num_values, double start, double end, int* status)
{
    if (*status) return;

    double inc = (end - start) / (double)num_values;
    if ((int)oskar_mem_length(values) != num_values)
    {
        oskar_mem_realloc(values, num_values, status);
        if (*status) return;
    }

    if (oskar_mem_type(values) == OSKAR_DOUBLE)
    {
        double* values_ = oskar_mem_double(values, status);
        for (int i = 0; i < num_values; ++i)
            values_[i] = start + i * inc;
    }
    else if (oskar_mem_type(values) == OSKAR_SINGLE)
    {
        float* values_ = oskar_mem_float(values, status);
        for (int i = 0; i < num_values; ++i)
            values_[i] = start + i * inc;
    }
    else
        *status = OSKAR_ERR_BAD_DATA_TYPE;
}
