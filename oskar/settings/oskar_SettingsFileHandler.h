/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_FILE_HANDLER_HPP_
#define OSKAR_SETTINGS_FILE_HANDLER_HPP_

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

namespace oskar {

struct SettingsFileHandlerPrivate;
class SettingsTree;

/**
 * @class SettingsFileHandler
 *
 * @brief Base class for settings file I/O classes.
 */
class OSKAR_SETTINGS_EXPORT SettingsFileHandler
{
public:
    SettingsFileHandler(const char* app, const char* version);
    virtual ~SettingsFileHandler();

    const char* app() const;
    const char* version() const;
    const char* file_name() const;
    virtual char* read(const char* file_name, const char* key) const = 0;
    virtual bool read_all(SettingsTree* tree) = 0;
    virtual bool write_all(const SettingsTree* tree) = 0;
    void set_file_name(const char* name);
    void set_write_defaults(bool value);
    bool write_defaults() const;

private:
    SettingsFileHandlerPrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_FILE_HANDLER_HPP_ */
