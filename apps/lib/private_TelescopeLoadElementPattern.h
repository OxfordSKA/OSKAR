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

#ifndef OSKAR_TELESCOPE_LOAD_ELEMENT_PATTERN_H_
#define OSKAR_TELESCOPE_LOAD_ELEMENT_PATTERN_H_

#include "apps/lib/oskar_TelescopeLoadAbstract.h"

struct oskar_Settings;
struct oskar_Log;

#include <vector>

class TelescopeLoadElementPattern : public oskar_TelescopeLoadAbstract
{
public:
    TelescopeLoadElementPattern(const oskar_Settings* settings,
            oskar_Log* log);

    virtual ~TelescopeLoadElementPattern();

    virtual void load(oskar_Telescope* telescope, const oskar_Dir& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);

    virtual void load(oskar_Station* station, const oskar_Dir& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);

    virtual std::string name() const;

private:
    double frequency_from_filename(const std::string& filename, int* status);
    int index_from_filename(const std::string& filename, int* status);

    void load_element_patterns(oskar_Station* station,
            const std::map<std::string, std::string>& filemap, int* status);
    void load(int port, oskar_Station* station,
            const std::vector<std::string>& keys,
            const std::vector<std::string>& paths, int* status);

    void update_map(std::map<std::string, std::string>& files,
            const oskar_Dir& cwd);

private:
    static const std::string root_name;
    std::string root_x;
    std::string root_y;
    std::string root_scalar;
    const oskar_Settings* settings_;
    oskar_Log* log_;
    std::map<std::string, int> models;
};

#endif /* OSKAR_TELESCOPE_LOAD_ELEMENT_PATTERN_H_ */
