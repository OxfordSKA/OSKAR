/*
 * Copyright (c) 2017-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsFileHandler.h"
#include <string>

using std::string;

namespace oskar {

struct SettingsFileHandlerPrivate
{
    bool write_defaults;
    string app;
    string version;
    string file_name;
};

SettingsFileHandler::SettingsFileHandler(const char* app, const char* version)
{
    p = new SettingsFileHandlerPrivate;
    p->write_defaults = false;
    p->app = string(app);
    p->version = string(version);
}

SettingsFileHandler::~SettingsFileHandler()
{
    delete p;
}

const char* SettingsFileHandler::app() const
{
    return p->app.c_str();
}

const char* SettingsFileHandler::version() const
{
    return p->version.c_str();
}

void SettingsFileHandler::set_file_name(const char* name)
{
    if (!name) return;
    p->file_name = string(name);
}

const char* SettingsFileHandler::file_name() const
{
    return p->file_name.c_str();
}

void SettingsFileHandler::set_write_defaults(bool value)
{
    p->write_defaults = value;
}

bool SettingsFileHandler::write_defaults() const
{
    return p->write_defaults;
}

} // namespace oskar
