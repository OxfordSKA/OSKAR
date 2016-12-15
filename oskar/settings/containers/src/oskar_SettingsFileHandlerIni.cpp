/*
 * Copyright (c) 2016, The University of Oxford
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

#include "oskar_SettingsFileHandlerIni.hpp"
#include <iostream>
#include <cstdio>
#include <fstream>
#include <sstream>

using namespace std;

namespace oskar {

struct SettingsFileHandlerIniPrivate
{
    ofstream file;
    string group;
};

SettingsFileHandlerIni::SettingsFileHandlerIni(
        const std::string& version)
: SettingsFileHandler(version)
{
    p = new SettingsFileHandlerIniPrivate;
}

SettingsFileHandlerIni::~SettingsFileHandlerIni()
{
    delete p;
}

string SettingsFileHandlerIni::trim(const std::string& s,
        const std::string& whitespace)
{
    size_t i0 = s.find_first_not_of(whitespace);
    if (i0 == std::string::npos) return "";
    size_t i1 = s.find_last_not_of(whitespace);
    return s.substr(i0, i1 - i0 + 1);
}

bool SettingsFileHandlerIni::read_all(SettingsTree* tree,
        vector<pair<string, string> >& invalid)
{
    if (filename_.empty()) return false;

    // Open the file.
    ifstream file;
    file.open(filename_.c_str());
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

            // Set key separators to forward slashes.
            size_t i, len;
            len = k.size();
            for (i = 0; i < len; ++i)
                if (k[i] == '\\' || k[i] == '.') k[i] = '/';

            // Try to set the item, and record if it fails.
            if (!group.empty() && group != "General")
                k = group + '/' + k;
            else if (k == "version")
                continue;
            if (!tree->set_value(k, v, false))
                invalid.push_back(make_pair(k, v));
        }
        else if (!line.empty())
        {
            // Set the current group.
            group = trim(line, "[]()");
        }
    }
    file.close();
    return true;
}

bool SettingsFileHandlerIni::write_all(const SettingsTree* tree)
{
    if (filename_.empty()) return false;

    // Open the file.
    p->file.open(filename_.c_str(), ofstream::trunc);
    if (!p->file) return false;

    // Recursively write from the root node.
    p->group = "";
    write_version();
    write(tree->root_node());
    p->file.close();
    return true;
}

void SettingsFileHandlerIni::set_file_name(const string& name)
{
    filename_ = name;
}

string SettingsFileHandlerIni::file_version() const
{
    string version;
    if (filename_.empty()) return string();

    // Open the file.
    ifstream file;
    file.open(filename_.c_str());
    if (!file) return string();

    // Loop over each line.
    for (string line; getline(file, line);)
    {
        line = trim(line, " \t");

        // Check if this is a key.
        if (line.find('=') != string::npos)
        {
            string k, v;
            stringstream ss;
            ss.str(line);
            getline(ss, k, '=');
            getline(ss, v);
            if (k == "version")
            {
                version = v;
                break;
            }
        }
    }
    file.close();
    return version;
}

void SettingsFileHandlerIni::write(const SettingsNode* node)
{
    const SettingsValue& val = node->value();
    string k = string(node->key());
    if (val.type() != SettingsValue::UNDEF)
    {
        if (node->value_or_child_set() || write_defaults_)
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
            p->file << k << "=" << val.to_string() << endl;
        }
    }
    for (int i = 0; i < node->num_children(); ++i)
        write(node->child(i));
}

void SettingsFileHandlerIni::write_version()
{
    p->file << "[General]" << endl;
    p->file << "version=" << version_ << endl;
}

} // namespace oskar
