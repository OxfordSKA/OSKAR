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

#ifndef OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_
#define OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_

#include <telescope/oskar_TelescopeLoadAbstract.h>

#include <vector>

class TelescopeLoaderElementPattern : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoaderElementPattern();
    virtual ~TelescopeLoaderElementPattern();
    virtual void load(oskar_Telescope* telescope, const std::string& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);
    virtual void load(oskar_Station* station, const std::string& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);
    virtual std::string name() const;

private:
    void load_element_patterns(oskar_Station* station,
            const std::map<std::string, std::string>& filemap, int* status);
    void load_fitted_data(int port, oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);
    void load_functional_data(int port, oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);
    static void parse_filename(const char* s, char** buffer, size_t* buflen,
            int* index, double* freq);
    void update_map(std::map<std::string, std::string>& files,
            const std::string& cwd);

private:
    std::string wildcard;
    std::string fit_root_x;
    std::string fit_root_y;
    std::string fit_root_scalar;
    std::string root_x;
    std::string root_y;
    oskar_Telescope* telescope_;
};

#endif /* OSKAR_TELESCOPE_LOADER_ELEMENT_PATTERN_H_ */
