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

#ifndef OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_HPP_
#define OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_HPP_

#ifdef __cplusplus

#include <oskar_SettingsFileHandler.hpp>

namespace oskar {

struct SettingsFileHandlerQSettingsPrivate;

/**
 * @class SettingsFileHandlerQSettings
 *
 * @brief Settings file interface using QSettings.
 *
 * @details
 * Read of the ini file
 *  - Creates a QSettings object for the ini file.
 *  - Loop though the settings defined in the tree and attempts settings.
 *
 * Write of the ini file.
 * - Loop though settings in the tree and write.
 *
 * Requirements to interface with the tree.
 * - When a setting is modified in the tree write it to the ini file.
 * - When the ini file is modified update the settings tree.
 *
 * SettingsTree tree;
 * tree.set_ini_file(ini_file)
 * tree.define_settings(xml_string);
 *
 *
 */
class SettingsFileHandlerQSettings : public oskar::SettingsFileHandler
{
public:
    SettingsFileHandlerQSettings(const std::string& version = std::string());
    virtual ~SettingsFileHandlerQSettings();

    virtual bool read_all(SettingsTree* tree,
            std::vector<std::pair<std::string, std::string> >& failed);
    virtual bool write_all(const SettingsTree* tree);
    virtual void set_file_name(const std::string& name);
    virtual std::string file_version() const;

private:
    SettingsFileHandlerQSettings(const SettingsFileHandlerQSettings&);
    const SettingsFileHandlerQSettings& operator=(
            const SettingsFileHandlerQSettings&);
    void write(const SettingsNode* item);
    void write_version();

private:
    SettingsFileHandlerQSettingsPrivate* p;
};

} /* namespace oskar */

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* C interface. */
struct oskar_SettingsFileHandlerQSettings;
#ifndef OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_TYPEDEF_
#define OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_TYPEDEF_
typedef struct oskar_SettingsFileHandlerQSettings oskar_SettingsFileHandlerQSettings;
#endif /* OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_TYPEDEF_ */

oskar_SettingsFileHandlerQSettings*
oskar_settings_file_handler_qsettings_create();

void oskar_settings_file_handler_qsettings_free(
        oskar_SettingsFileHandlerQSettings* handler);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_FILE_HANDLER_QSETTINGS_HPP_ */
