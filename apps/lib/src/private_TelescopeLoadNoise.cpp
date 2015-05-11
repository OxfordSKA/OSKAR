/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "apps/lib/private_TelescopeLoadNoise.h"
#include "apps/lib/oskar_dir.h"

#include <oskar_Settings.h>
#include <oskar_file_exists.h>

#include <cfloat>
#include <cassert>
#include <cmath>

using std::map;
using std::string;

TelescopeLoadNoise::TelescopeLoadNoise(const oskar_Settings* settings)
: oskar_TelescopeLoadAbstract(), dataType_(0), freqs_(0)
{
    files_[FREQ] = "noise_frequencies.txt";
    files_[RMS]  = "rms.txt";
    settings_ = settings;
}

TelescopeLoadNoise::~TelescopeLoadNoise()
{
    int status = 0;
    oskar_mem_free(freqs_, &status);
}


// Depth = 0
// - Set up frequency data as this is the same for all stations
//   and if defined by files these have to be at depth 0.
void TelescopeLoadNoise::load(oskar_Telescope* telescope, const oskar_Dir& cwd,
        int num_subdirs, map<string, string>& filemap, int* status)
{
    if (*status || !settings_->interferometer.noise.enable)
        return;

    dataType_ = oskar_telescope_precision(telescope);

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);

    // Set up noise frequency values (this only happens at depth = 0)
    freqs_ = oskar_mem_create(dataType_, OSKAR_CPU, 0, status);
    getNoiseFreqs_(freqs_, filemap[files_[FREQ]], status);

    // If no sub-directories (the station load function is never called)
    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_stations(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* s = oskar_telescope_station(telescope, i);
            oskar_mem_copy(oskar_station_noise_freq_hz(s), freqs_, status);
            setNoiseRMS_(s, filemap, status);
        }
    }
}


// Depth > 0
void TelescopeLoadNoise::load(oskar_Station* station,
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
        /**status = OSKAR_ERR_SETTINGS_INTERFEROMETER_NOISE; */
        return;
    }

    // Update the noise files for the current station directory.
    updateFileMap_(filemap, cwd);

    // Set the frequency noise data field of the station structure.
    oskar_mem_copy(oskar_station_noise_freq_hz(station), freqs_, status);

    // Set the noise RMS based on files or settings.
    setNoiseRMS_(station, filemap, status);
}

string TelescopeLoadNoise::name() const
{
    return string("noise loader");
}

// -- private functions -------------------------------------------------------

void TelescopeLoadNoise::updateFileMap_(map<string, string>& filemap,
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

void TelescopeLoadNoise::getNoiseFreqs_(oskar_Mem* freqs,
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
        oskar_mem_load_ascii(filename.c_str(), 1, status, freqs, "");
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

void TelescopeLoadNoise::setNoiseRMS_(oskar_Station* model,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;
    oskar_Mem* noise_rms = oskar_station_noise_rms_jy(model);
    int num_freqs = (int)oskar_mem_length(oskar_station_noise_freq_hz(model));
    noiseSpecRMS_(noise_rms, num_freqs, filemap, status);
}


void TelescopeLoadNoise::noiseSpecRMS_(oskar_Mem* rms, int num_freqs,
        const map<string, string>& filemap, int* status)
{
    if (*status) return;

    const oskar_SettingsSystemNoiseRMS& settingsRMS =
            settings_->interferometer.noise.rms;

    string filename;
    if (filemap.count(files_[RMS]))
        filename = filemap.at(files_[RMS]);

    switch (settingsRMS.specification)
    {
        case OSKAR_SYSTEM_NOISE_TELESCOPE_MODEL:
            oskar_mem_load_ascii(filename.c_str(), 1, status, rms, "");
            break;
        case OSKAR_SYSTEM_NOISE_DATA_FILE:
            oskar_mem_load_ascii(settingsRMS.file, 1, status, rms, "");
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


void TelescopeLoadNoise::evaluate_range_(oskar_Mem* values,
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
