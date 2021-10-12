/*
 * Copyright (c) 2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderStationTypeMap.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* station_type_map_file = "station_type_map.txt";

void TelescopeLoaderStationTypeMap::load(oskar_Telescope* telescope,
        const string& cwd, int /*num_subdirs*/, map<string, string>& /*filemap*/,
        int* status)
{
    // Check for presence of "station_type_map.txt".
    if (oskar_dir_file_exists(cwd.c_str(), station_type_map_file))
    {
        oskar_telescope_load_station_type_map(telescope,
                get_path(cwd, station_type_map_file).c_str(), status);
    }
}

string TelescopeLoaderStationTypeMap::name() const
{
    return string("station type map file loader");
}
