/*
 * Copyright (c) 2016-2017, The University of Oxford
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

#include "settings/oskar_SettingsFileHandlerIni.h"
#include "settings/oskar_SettingsDependency.h"
#include "settings/oskar_SettingsNode.h"
#include "settings/oskar_SettingsTree.h"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

using namespace std;

namespace oskar {

static string trim(const string& s, const string& whitespace)
{
    size_t i0 = s.find_first_not_of(whitespace);
    if (i0 == std::string::npos) return "";
    size_t i1 = s.find_last_not_of(whitespace);
    return s.substr(i0, i1 - i0 + 1);
}

struct SettingsFileHandlerIniPrivate
{
    ofstream file;
    string group;
};

SettingsFileHandlerIni::SettingsFileHandlerIni(
        const char* app, const char* version)
: SettingsFileHandler(app, version)
{
    p = new SettingsFileHandlerIniPrivate;
}

SettingsFileHandlerIni::~SettingsFileHandlerIni()
{
    delete p;
}

char* SettingsFileHandlerIni::read(const char* file_name,
        const char* key) const
{
    // Open the file.
    if (!file_name || strlen(file_name) == 0) return 0;
    ifstream file;
    file.open(file_name);
    if (!file) return 0;

    // Loop over each line.
    string value;
    string key_ = string(key);
    for (string line, group, k; getline(file, line);)
    {
        line = trim(line, " \t");

        // Check if this is a key.
        if (line.find('=') != string::npos)
        {
            stringstream ss;
            ss.str(line);
            getline(ss, k, '=');

            // Prepend group to key.
            if (!group.empty() && group != "General")
                k = group + '/' + k;

            // Set key separators to forward slashes.
            size_t i, len;
            len = k.size();
            for (i = 0; i < len; ++i)
                if (k[i] == '\\' || k[i] == '.') k[i] = '/';

            // Check if we have found the required key.
            if (k == key_)
            {
                getline(ss, value);
                break;
            }
        }
        else if (!line.empty() && line[0] == '[')
        {
            // Set the current group.
            group = trim(line, "[]");
        }
    }
    file.close();

    // Copy the string into a char array and return it.
    char* buffer = (char*) calloc(1 + value.size(), sizeof(char));
    if (value.size() > 0) memcpy(buffer, value.c_str(), value.size());
    return buffer;
}

bool SettingsFileHandlerIni::read_all(SettingsTree* tree)
{
    if (!file_name() || strlen(file_name()) == 0) return false;

    // Open the file.
    ifstream file;
    file.open(file_name());
    if (!file) return false;

    // Loop over each line.
    for (string line, group, k, v; getline(file, line);)
    {
        line = trim(line, " \t");

        // Check if this is a key.
        if (line.find('=') != string::npos)
        {
            stringstream ss;
            ss.str(line);
            getline(ss, k, '=');
            getline(ss, v);

            // Prepend group to key.
            if (!group.empty() && group != "General")
                k = group + '/' + k;

            // Set key separators to forward slashes.
            size_t i, len;
            len = k.size();
            for (i = 0; i < len; ++i)
                if (k[i] == '\\' || k[i] == '.') k[i] = '/';

            // Try to set the item, and record if it fails.
            if (k == "version" || k == "app")
                continue;

            if (!tree->set_value(k.c_str(), v.c_str(), false))
                tree->add_failed(k.c_str(), v.c_str());
        }
        else if (!line.empty() && line[0] == '[')
        {
            // Set the current group.
            group = trim(line, "[]");
        }
    }
    file.close();
    return true;
}

bool SettingsFileHandlerIni::write_all(const SettingsTree* tree)
{
    if (!file_name() || strlen(file_name()) == 0) return false;

    // Open the file.
    p->file.open(file_name(), ofstream::trunc);
    if (!p->file) return false;

    // Recursively write from the root node.
    p->group = "";
    write_header();
    write(tree->root_node());
    p->file.close();
    return true;
}

void SettingsFileHandlerIni::write(const SettingsNode* node)
{
    string k = string(node->key());
    if (node->item_type() == SettingsItem::SETTING)
    {
        if (node->value_or_child_set() || this->write_defaults())
        {
            // Find the top-level group.
            size_t i = k.find_first_of("/\\.");
            if (i != string::npos)
            {
                string group = k.substr(0, i);
                if (group != p->group)
                {
                    p->group = group;
                    p->file << endl << "[" << group << "]" << endl;
                }
                k = k.substr(i + 1);
            }
            p->file << k << "=" << node->value() << endl;
        }
    }
    for (int i = 0; i < node->num_children(); ++i)
        write(node->child(i));
}

void SettingsFileHandlerIni::write_header()
{
    p->file << "[General]" << endl;
    p->file << "app=" << this->app() << endl;
    p->file << "version=" << this->version() << endl;
}

} // namespace oskar
