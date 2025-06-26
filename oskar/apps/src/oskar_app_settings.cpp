/*
 * Copyright (c) 2017-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "apps/oskar_app_settings.h"
#include "apps/oskar_app_registrar.h"
#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "oskar_version.h"
#include <cstring>
#include <cstdio>

using std::map;
using std::string;
using namespace oskar;

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
            // NOLINTNEXTLINE: clang-tidy doesn't realise this calls 'delete'.
            SettingsTree::free(s);
            return 0;
        }
    }
    return s;
}
