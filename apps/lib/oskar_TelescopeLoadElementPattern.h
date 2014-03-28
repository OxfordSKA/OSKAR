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

/**
 * @file oskar_TelescopeLoadElementPattern.h
 */

#include "apps/lib/oskar_TelescopeLoadAbstract.h"

struct oskar_Settings;
struct oskar_Log;

class OSKAR_APPS_EXPORT oskar_TelescopeLoadElementPattern
: public oskar_TelescopeLoadAbstract
{
public:
    oskar_TelescopeLoadElementPattern(const oskar_Settings* settings,
            oskar_Log* log);

    virtual ~oskar_TelescopeLoadElementPattern();

    /**
     * @brief
     * Loads data into the top-level telescope model structure.
     *
     * @details
     * Implement this function to load a data file into the top-level telescope
     * model.
     *
     * @param[in,out] telescope Pointer to telescope model.
     * @param[in] cwd Reference to the current working directory.
     * @param[in] num_subdirs Number of subdirectories in the working directory.
     * @param[in,out] filemap Reference to file map to use for this level.
     *                        This should be updated for use at a deeper
     *                        level if necessary.
     * @param[in,out] status Status return code.
     */
    virtual void load(oskar_Telescope* telescope, const oskar_Dir& cwd,
            int num_subdirs, std::map<std::string, std::string>& filemap,
            int* status);

    /**
     * @brief
     * Loads data into a station model structure.
     *
     * @details
     * Implement this function to load a data file into a station model.
     *
     * @param[in,out] station Pointer to station model.
     * @param[in] cwd Reference to the current working directory.
     * @param[in] num_subdirs Number of subdirectories in the working directory.
     * @param[in] depth Current depth index.
     * @param[in,out] filemap Reference to file map to use for this level.
     *                        This should be updated for use at a deeper
     *                        level if necessary.
     * @param[in,out] status Status return code.
     */
    virtual void load(oskar_Station* station, const oskar_Dir& cwd,
            int num_subdirs, int depth,
            std::map<std::string, std::string>& filemap, int* status);

    /**
     * @brief
     * Returns a readable name for the loader.
     */
    virtual std::string name() const;

private:
    double frequency_from_filename(const std::string& filename, int startidx,
            int* status);

    void load_element_patterns(oskar_Station* station,
            const std::map<std::string, std::string>& filemap, int* status);

    void update_map(std::map<std::string, std::string>& files,
            const oskar_Dir& cwd);

private:
    static const std::string root_name;
    std::string root_x;
    std::string root_y;
    const oskar_Settings* settings_;
    oskar_Log* log_;
    std::map<std::string, int> models;
};

#endif /* OSKAR_TELESCOPE_LOAD_ELEMENT_PATTERN_H_ */
