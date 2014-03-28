/*
 * Copyright (c) 2013, The University of Oxford
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

#include "apps/lib/oskar_TelescopeLoadLayout.h"
#include "apps/lib/oskar_Dir.h"
#include <oskar_Settings.h>

using std::map;
using std::string;

const string oskar_TelescopeLoadLayout::layout_file = "layout.txt";
const string oskar_TelescopeLoadLayout::layout_ecef_file = "layout_ecef.txt";

oskar_TelescopeLoadLayout::oskar_TelescopeLoadLayout(
        const oskar_Settings* settings)
{
    settings_ = settings;
}

oskar_TelescopeLoadLayout::~oskar_TelescopeLoadLayout()
{
}

void oskar_TelescopeLoadLayout::load(oskar_Telescope* telescope,
        const oskar_Dir& cwd, int num_subdirs,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of top-level "layout.txt" or "layout_ecef.txt".
    if (cwd.exists(layout_file))
    {
        // Load the interferometer layout (horizon plane).
        oskar_telescope_load_station_coords_horizon(telescope,
                cwd.absoluteFilePath(layout_file).c_str(),
                settings_->telescope.longitude_rad,
                settings_->telescope.latitude_rad,
                settings_->telescope.altitude_m, status);
    }
    else if (cwd.exists(layout_ecef_file))
    {
        // Load the interferometer layout (ECEF system).
        oskar_telescope_load_station_coords_ecef(telescope,
                cwd.absoluteFilePath(layout_ecef_file).c_str(),
                settings_->telescope.longitude_rad,
                settings_->telescope.latitude_rad,
                settings_->telescope.altitude_m, status);
    }
    else
    {
        // If telescope hasn't already been sized using a (deprecated)
        // "config.txt" file, return an error.
        if (oskar_telescope_num_stations(telescope) == 0)
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;
        return;
    }

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

void oskar_TelescopeLoadLayout::load(oskar_Station* station,
        const oskar_Dir& cwd, int num_subdirs, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of "layout.txt".
    if (cwd.exists(layout_file))
    {
        oskar_station_load_layout(station,
                cwd.absoluteFilePath(layout_file).c_str(), status);
    }
    else
    {
        // If station hasn't already been sized using a (deprecated)
        // "config.txt" file, return an error.
        if (oskar_station_num_elements(station) == 0)
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;
        return;
    }

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

string oskar_TelescopeLoadLayout::name() const
{
    return string("layout file loader");
}
