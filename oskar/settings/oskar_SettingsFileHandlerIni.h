/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_FILE_HANDLER_INI_HPP_
#define OSKAR_SETTINGS_FILE_HANDLER_INI_HPP_

#ifdef __cplusplus

#include <settings/oskar_SettingsFileHandler.h>

namespace oskar {

struct SettingsFileHandlerIniPrivate;
class SettingsTree;
class SettingsNode;

/**
 * @class SettingsFileHandlerIni
 *
 * @brief Settings file interface.
 *
 * @details
 * Read of the ini file
 *  - Loop though the settings defined in the tree and attempts settings.
 *
 * Write of the ini file.
 * - Loop though settings in the tree and write.
 */
class OSKAR_SETTINGS_EXPORT SettingsFileHandlerIni
: public SettingsFileHandler
{
public:
    SettingsFileHandlerIni(const char* app, const char* version);
    virtual ~SettingsFileHandlerIni();

    virtual char* read(const char* file_name, const char* key) const;
    virtual bool read_all(SettingsTree* tree);
    virtual bool write_all(const SettingsTree* tree);

private:
    SettingsFileHandlerIni(const SettingsFileHandlerIni&);
    const SettingsFileHandlerIni& operator=(
            const SettingsFileHandlerIni&);
    void write(const SettingsNode* item);
    void write_header();

private:
    SettingsFileHandlerIniPrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_FILE_HANDLER_INI_HPP_ */
