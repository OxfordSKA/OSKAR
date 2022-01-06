/*
 * Copyright (c) 2017, The University of Oxford
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

#include "apps/oskar_app_settings.h"
#include "apps/oskar_app_registrar.h"
#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "oskar_version.h"
#include <cstring>
#include <cstdio>

using namespace oskar;
using namespace std;

const char* oskar_app_settings(const char* app)
{
    map<string, const char*>::iterator it =
            AppRegistrar::apps().find(string(app));
    return (it != AppRegistrar::apps().end()) ? it->second : 0;
}

SettingsTree* oskar_app_settings_tree(const char* app,
        const char* settings_file)
{
    // Get the settings.
    const char* settings_str = oskar_app_settings(app);
    if (!settings_str) return 0;

    // Create the settings tree and declare the settings.
    SettingsTree* s = new SettingsTree;
    oskar_settings_declare_xml(s, settings_str);

    // If a filename is given, try to load it.
    if (settings_file && strlen(settings_file) > 0)
    {
        SettingsFileHandlerIni* handler = new SettingsFileHandlerIni(
                app, OSKAR_VERSION_STR);
        s->set_file_handler(handler);
        if (!s->load(settings_file))
        {
            SettingsTree::free(s);
            return 0;
        }
    }
    return s;
}
