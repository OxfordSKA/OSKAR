/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_LOAD_NOISE_H_
#define OSKAR_TELESCOPE_LOAD_NOISE_H_

#include "apps/lib/oskar_TelescopeLoadAbstract.h"

struct oskar_Settings;

class TelescopeLoadNoise : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoadNoise(const oskar_Settings* settings);
    ~TelescopeLoadNoise();

    void load(oskar_Telescope* telescope, const oskar_Dir& cwd, int num_subdirs,
            std::map<std::string, std::string>& filemap, int* status);

    void load(oskar_Station* station, const oskar_Dir& cwd, int num_subdirs,
            int depth, std::map<std::string, std::string>& filemap,
            int* status);

    virtual std::string name() const;

private:
    // Updates set of files to load.
    void updateFileMap_(std::map<std::string, std::string>& filemap,
            const oskar_Dir& cwd);

    // Obtains the array of noise frequencies.
    void getNoiseFreqs_(oskar_Mem* freqs, const std::string& filepath,
            int* status);

    // Obtains the noise RMS values and sets then into the telescope model.
    void setNoiseRMS_(oskar_Station* model,
            const std::map<std::string, std::string>& filemap, int* status);

    // Obtains noise RMS values for telescope model priority loading.
    void noiseSpecTelescopeModel_(oskar_Mem* rms, int num_freqs,
            double bandwidth_hz, double integration_time_sec,
            const std::map<std::string, std::string>& filemap, int* status);

    // Obtains noise RMS values for RMS specification.
    void noiseSpecRMS_(oskar_Mem* rms, int num_freqs,
            const std::map<std::string, std::string>& filemap, int* status);

    // Obtains noise RMS values for sensitivity specification.
    void noiseSpecSensitivity_(oskar_Mem* rms, int num_freqs,
            double bandwidth_hz, double integration_time_sec,
            const std::map<std::string, std::string>& filemap, int* status);

    // Obtains noise RMS values for Tsys specification.
    void noiseSpecTsys_(oskar_Mem* rms, int num_freqs,
            double bandwidth_hz, double integration_time_sec,
            const std::map<std::string, std::string>& filemap, int* status);

    void sensitivity_to_rms_(oskar_Mem* rms, const oskar_Mem* sensitivity,
            int num_freqs, double bandwidth_hz, double integration_time_sec,
            int* status);

    void t_sys_to_rms_(oskar_Mem* rms, const oskar_Mem* t_sys,
            const oskar_Mem* area, const oskar_Mem* efficiency, int num_freqs,
            double bandwidth, double integration_time, int* status);

    void evaluate_range_(oskar_Mem* values, int num_values, double start,
            double end, int* status);

private:
    enum FileIds_ { FREQ, RMS, SENSITIVITY, TSYS, AREA, EFFICIENCY };
    int dataType_;  // OSKAR data type of the telescope model being loaded.
    std::map<FileIds_, std::string> files_;
    oskar_Mem* freqs_;
    const oskar_Settings* settings_;
};

#endif /* OSKAR_TELESCOPE_LOAD_NOISE_H_ */
