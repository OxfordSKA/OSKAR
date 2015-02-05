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

#include "apps/lib/private_TelescopeLoadConfig.h"
#include "apps/lib/oskar_dir.h"
#include <oskar_Settings.h>

using std::map;
using std::string;

const string TelescopeLoadConfig::config_file = "config.txt";

TelescopeLoadConfig::TelescopeLoadConfig(const oskar_Settings* settings)
{
    settings_ = settings;
}

TelescopeLoadConfig::~TelescopeLoadConfig()
{
}

void TelescopeLoadConfig::load(oskar_Telescope* telescope,
        const oskar_Dir& cwd, int num_subdirs,
        map<string, string>& /*filemap*/, int* status)
{
    // Return immediately if "config.txt" doesn't exist.
    // Not a problem for the moment - "layout.txt" may exist, so we can
    // check for that in the next loader.
    if (!cwd.exists(config_file))
        return;

    // Load the interferometer layout.
    oskar_telescope_load_station_coords_horizon(telescope,
            cwd.absoluteFilePath(config_file).c_str(),
            settings_->telescope.longitude_rad,
            settings_->telescope.latitude_rad,
            settings_->telescope.altitude_m, status);

    // If no subdirectories, set all "stations" to be a single dipole.
    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_stations(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* station = oskar_telescope_station(telescope, i);
            oskar_station_resize(station, 1, status);
            oskar_station_resize_element_types(station, 1, status);
            oskar_station_set_element_coords(station, 0,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
            oskar_station_set_element_errors(station, 0,
                    1.0, 0.0, 0.0, 0.0, status);
            oskar_station_set_element_orientation(station, 0,
                    90.0, 0.0, status);
            oskar_station_set_element_weight(station, 0, 1.0, 0.0, status);
        }
    }
}

void TelescopeLoadConfig::load(oskar_Station* station,
        const oskar_Dir& cwd, int num_subdirs, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Return immediately if "config.txt" doesn't exist.
    // Not a problem for the moment - "layout.txt" may exist, so we can
    // check for that in the next loader.
    if (!cwd.exists(config_file))
        return;

    // Load the station configuration.
    oskar_station_load_config(station,
            cwd.absoluteFilePath(config_file).c_str(), status);

    // Check if this is the last level.
    if (num_subdirs > 0)
    {
        // Allocate storage for child stations.
        oskar_station_create_child_stations(station, status);
    }
    else if (num_subdirs == 0)
    {
        // Allocate storage for element model data if no children.
        oskar_station_resize_element_types(station, 1, status);
    }
}

string TelescopeLoadConfig::name() const
{
    return string("config file loader");
}
