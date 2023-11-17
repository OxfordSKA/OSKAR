/*
 * Copyright (c) 2023, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderVirtualAntennaAngle.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* antenna_angle_file = "virtual_antenna_angle.txt";

void TelescopeLoaderVirtualAntennaAngle::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of antenna angle file.
    if (oskar_dir_file_exists(cwd.c_str(), antenna_angle_file))
    {
        string f = get_path(cwd, antenna_angle_file);
        oskar_station_load_virtual_antenna_angle(
                station, f.c_str(), status
        );
    }
}

void TelescopeLoaderVirtualAntennaAngle::load(oskar_Telescope* telescope,
        const string& cwd, int /*num_subdirs*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of antenna angle file.
    if (oskar_dir_file_exists(cwd.c_str(), antenna_angle_file))
    {
        string f = get_path(cwd, antenna_angle_file);
        oskar_telescope_load_virtual_antenna_angle(
                telescope, f.c_str(), status
        );
    }
}

string TelescopeLoaderVirtualAntennaAngle::name() const
{
    return string("virtual antenna angle file loader");
}
