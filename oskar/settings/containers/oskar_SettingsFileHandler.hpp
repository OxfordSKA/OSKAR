/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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

#ifndef OSKAR_SETTINGS_FILE_HANDLER_HPP_
#define OSKAR_SETTINGS_FILE_HANDLER_HPP_

#include <oskar_SettingsTree.hpp>

#ifdef __cplusplus

#include <string>
#include <utility>
#include <vector>

namespace oskar {

/**
 * @class SettingsFileHandler
 *
 * @brief Base class for settings file I/O classes.
 */
class SettingsFileHandler
{
 public:
    SettingsFileHandler(const std::string& version = std::string()) :
        write_defaults_(false), version_(version) {}
    virtual ~SettingsFileHandler() {}

    void set_write_defaults(bool value) { write_defaults_ = value; }
    virtual bool read_all(SettingsTree* tree,
            std::vector<std::pair<std::string, std::string> >& invalid) = 0;
    virtual bool write_all(const SettingsTree* tree) = 0;
    virtual void set_file_name(const std::string& name) = 0;
    virtual std::string file_version() const = 0;

 protected:
    bool write_defaults_;
    std::string version_;
    std::string file_version_;
    std::string filename_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_FILE_HANDLER_HPP_ */
