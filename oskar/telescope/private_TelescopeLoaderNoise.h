/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#ifndef OSKAR_TELESCOPE_LOADER_NOISE_H_
#define OSKAR_TELESCOPE_LOADER_NOISE_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

class TelescopeLoaderNoise : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderNoise();
    virtual ~TelescopeLoaderNoise();
    void load(oskar_Telescope* telescope, const std::string& cwd, int num_subdirs,
            std::map<std::string, std::string>& filemap, int* status);
    void load(oskar_Station* station, const std::string& cwd, int num_subdirs,
            int depth, std::map<std::string, std::string>& filemap,
            int* status);
    virtual std::string name() const;

private:
    // Updates set of files to load.
    void update_map(std::map<std::string, std::string>& filemap,
            const std::string& cwd);

    // Obtains the noise RMS values and sets then into the telescope model.
    void set_noise_rms(oskar_Station* model,
            const std::map<std::string, std::string>& filemap, int* status);

private:
    enum FileIds_ { FREQ, RMS };
    oskar_Mem* freqs_;
    oskar_Telescope* telescope_;
    std::map<FileIds_, std::string> files_;
};

#endif /* OSKAR_TELESCOPE_LOADER_NOISE_H_ */
