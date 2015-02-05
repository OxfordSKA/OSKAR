/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "apps/lib/private_TelescopeLoadElementPattern.h"
#include "apps/lib/oskar_dir.h"
#include <oskar_log.h>
#include <oskar_file_exists.h>
#include <oskar_Settings.h>

#include <sstream>

using std::vector;
using std::map;
using std::string;

const string TelescopeLoadElementPattern::root_name =
        "element_pattern_fit_";

TelescopeLoadElementPattern::TelescopeLoadElementPattern(
        const oskar_Settings* settings, oskar_Log* log)
{
    settings_ = settings;
    log_ = log;
    root_x = root_name + "x_";
    root_y = root_name + "y_";
    root_scalar = root_name + "scalar_";
}

TelescopeLoadElementPattern::~TelescopeLoadElementPattern()
{
}

void TelescopeLoadElementPattern::load(oskar_Telescope* telescope,
        const oskar_Dir& cwd, int num_subdirs, map<string, string>& filemap,
        int* status)
{
    update_map(filemap, cwd);

    // Load element pattern data for stations only at the deepest level!
    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_stations(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* s = oskar_telescope_station(telescope, i);
            oskar_station_resize_element_types(s, 1, status);
            load_element_patterns(s, filemap, status);
        }
    }
}

void TelescopeLoadElementPattern::load(oskar_Station* station,
        const oskar_Dir& cwd, int num_subdirs, int /*depth*/,
        map<string, string>& filemap, int* status)
{
    update_map(filemap, cwd);

    // Load element pattern data for stations only at the deepest level!
    if (num_subdirs == 0)
    {
        oskar_station_resize_element_types(station, 1, status);
        load_element_patterns(station, filemap, status);
    }
}

string TelescopeLoadElementPattern::name() const
{
    return string("element pattern loader");
}

double TelescopeLoadElementPattern::frequency_from_filename(
        const string& filename, int* status)
{
    double freq = 0.0;
    size_t i1, i2;

    // Check if safe to proceed.
    if (*status) return 0.0;

    // Get indices of the delimiters.
    i2 = filename.find_last_of(".");
    if (i2 == string::npos)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0.0;
    }
    i2--;
    i1 = filename.find_last_of("_");
    if (i1 == string::npos)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0.0;
    }
    i1++;

    // Get the frequency part of the filename
    // (the last section before the file extension).
    string str = filename.substr(i1, 1+i2-i1);

    // Convert to number.
    std::stringstream ss(str);
    ss >> freq;
    if (!ss)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0.0;
    }
    return freq * 1e6; // Convert from MHz to Hz.
}

int TelescopeLoadElementPattern::index_from_filename(
        const string& filename, int* status)
{
    int ind = 0;
    size_t i1, i2;

    // Check if safe to proceed.
    if (*status) return 0;

    // Get indices of the delimiters.
    i2 = filename.find_last_of("_");
    if (i2 == string::npos)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    i2--;
    i1 = filename.find_last_of("_", i2); // Get the next "_".
    if (i1 == string::npos)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    i1++;

    // Get the index part of the filename
    // (the section between the last two underscores).
    string str = filename.substr(i1, 1+i2-i1);

    // Convert to number.
    std::stringstream ss(str);
    ss >> ind;
    if (!ss || ind < 0)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }
    return ind;
}

void TelescopeLoadElementPattern::load_element_patterns(
        oskar_Station* station, const map<string, string>& filemap, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Return if element patterns are disabled.
    if (!settings_->telescope.aperture_array.element_pattern.
            enable_numerical_patterns)
        return;

    // Get lists of all paths in the map that have keys starting with the
    // right root name.
    vector<string> keys_x, keys_y, keys_scalar;
    vector<string> paths_x, paths_y, paths_scalar;
    for (map<string, string>::const_iterator i = filemap.begin();
            i != filemap.end(); ++i)
    {
        string key = i->first;
        if (key.compare(0, root_x.size(), root_x) == 0)
        {
            keys_x.push_back(key);
            paths_x.push_back(i->second);
        }
        if (key.compare(0, root_y.size(), root_y) == 0)
        {
            keys_y.push_back(key);
            paths_y.push_back(i->second);
        }
        if (key.compare(0, root_scalar.size(), root_scalar) == 0)
        {
            keys_scalar.push_back(key);
            paths_scalar.push_back(i->second);
        }
    }

    // Load X, Y or scalar data.
    if (settings_->telescope.pol_mode == OSKAR_POL_MODE_FULL)
    {
        load(1, station, keys_x, paths_x, status);
        load(2, station, keys_y, paths_y, status);
    }
    else
        load(0, station, keys_scalar, paths_scalar, status);
}

void TelescopeLoadElementPattern::load(int port, oskar_Station* station,
        const vector<string>& keys, const vector<string>& paths, int* status)
{
    if (*status) return;
    const char* s = port == 0 ? "SCALAR" : port == 1 ? "X" : "Y";
    for (size_t i = 0; i < keys.size(); ++i)
    {
        string key = keys[i];
        string path = paths[i];

        // Get the element index and frequency from the key.
        int ind = index_from_filename(key, status);
        double freq = frequency_from_filename(key, status);

        // Load the file.
        if (models.count(path) == 0)
        {
            oskar_log_message(log_, 'M', 0,
                    "Loading fitted element pattern %s[%d] at %.0f MHz: %s",
                    s, ind, freq / 1.0e6, path.c_str());
            models[path] = 1;
        }
        if (*status) break;
        if (oskar_station_num_element_types(station) < ind + 1)
            oskar_station_resize_element_types(station, ind + 1, status);
        oskar_element_read(oskar_station_element(station, ind), path.c_str(),
                port, freq, status);
    }
}

void TelescopeLoadElementPattern::update_map(map<string, string>& filemap,
        const oskar_Dir& cwd)
{
    // Update the map of element files for the current directory.
    // The presence of fitted coefficients is sufficient to override ones
    // from a higher level.

    // Get a listing of the files in the current directory that start with
    // the fitted element data root name.
    vector<string> file_list = cwd.filesStartingWith(root_name);

    // Store the full paths to these files in the persistent map, with the
    // local filename as the key.
    for (size_t i = 0; i < file_list.size(); ++i)
    {
        filemap[file_list[i]] = cwd.absoluteFilePath(file_list[i]);
    }
}
