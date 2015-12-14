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

#include <oskar_SettingsFileHandlerQSettings.hpp>
#include <QtCore/QSettings>
#include <QtCore/QStringList>
#include <iostream>

using namespace std;

namespace oskar {

struct SettingsFileHandlerQSettingsPrivate
{
    QSettings* s;
    SettingsFileHandlerQSettingsPrivate() : s(0) {}
    ~SettingsFileHandlerQSettingsPrivate() { if (s) delete s; }
};

SettingsFileHandlerQSettings::SettingsFileHandlerQSettings(
        const std::string& version)
: SettingsFileHandler(version)
{
    p = new SettingsFileHandlerQSettingsPrivate;
}

SettingsFileHandlerQSettings::~SettingsFileHandlerQSettings()
{
    delete p;
}

bool SettingsFileHandlerQSettings::read_all(SettingsTree* tree,
        vector<pair<string, string> >& failed)
{
    if (!p || !p->s) return false;

    // Check that the file exists.
    FILE* file = fopen(filename_.c_str(), "r");
    if (!file) return false;
    fclose(file);

    // Read out all keys in the file.
    QStringList keys = p->s->allKeys();
    int num_keys = keys.size();
    for (int i = 0; i < num_keys; ++i)
    {
        QString key = keys[i];
        string k = key.toStdString();
        string v = p->s->value(key).toString().toStdString();
        if (k == "version") continue;

        // Try to set the item, and record if it fails.
        if (!tree->set_value(k, v, false))
            failed.push_back(make_pair(k, v));
    }
    return true;
}

bool SettingsFileHandlerQSettings::write_all(const SettingsTree* tree)
{
    if (!p || !p->s) return false;
    write_version();
    write(tree->root_node());
    return true;
}

void SettingsFileHandlerQSettings::set_file_name(const string& name)
{
    filename_ = name;
    if (p->s) delete p->s;
    p->s = new QSettings(QString::fromStdString(name), QSettings::IniFormat);
}

string SettingsFileHandlerQSettings::file_version() const
{
    if (!p->s) return string();
    return p->s->value("version").toString().toStdString();
}

void SettingsFileHandlerQSettings::write(const SettingsNode* node)
{
    const SettingsValue& val = node->value();
    QString k = QString::fromStdString(node->key());
    if (val.type() != SettingsValue::UNDEF)
    {
        if (node->value_or_child_set() || write_defaults_)
            p->s->setValue(k, QString::fromStdString(val.to_string()));
        else if (p->s->contains(k))
            p->s->remove(k);
    }
    for (int i = 0; i < node->num_children(); ++i)
        write(node->child(i));
}

void SettingsFileHandlerQSettings::write_version()
{
    // Write a version key only if it doesn't already exist in the file.
    if (p->s && !p->s->contains("version"))
        p->s->setValue("version", QString::fromStdString(version_));
}

} // namespace oskar

/* C interface. */
struct oskar_SettingsFileHandlerQSettings :
public oskar::SettingsFileHandlerQSettings
{
};

oskar_SettingsFileHandlerQSettings*
oskar_settings_file_handler_qsettings_create()
{
    return new oskar_SettingsFileHandlerQSettings;
}

void oskar_settings_file_handler_qsettings_free(
        oskar_SettingsFileHandlerQSettings* handler)
{
    delete handler;
}

