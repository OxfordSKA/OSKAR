/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderCableLengthError.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* cable_length_error_file = "cable_length_error.txt";
static const char* cable_length_error_x_file = "cable_length_error_x.txt";
static const char* cable_length_error_y_file = "cable_length_error_y.txt";

void TelescopeLoaderCableLengthError::load(
        oskar_Telescope* telescope,
        const string& cwd,
        int /*num_subdirs*/,
        map<string, string>& /*filemap*/,
        int* status
)
{
    // Check for presence of cable length error files.
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_file))
    {
        oskar_telescope_load_cable_length_error(telescope, 0,
                get_path(cwd, cable_length_error_file).c_str(), status);
        oskar_telescope_load_cable_length_error(telescope, 1,
                get_path(cwd, cable_length_error_file).c_str(), status);
}
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_x_file))
    {
        oskar_telescope_load_cable_length_error(telescope, 0,
                get_path(cwd, cable_length_error_x_file).c_str(), status);
    }
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_y_file))
    {
        oskar_telescope_load_cable_length_error(telescope, 1,
                get_path(cwd, cable_length_error_y_file).c_str(), status);
    }
}

void TelescopeLoaderCableLengthError::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of cable length error files.
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_file))
    {
        oskar_station_load_cable_length_error(station, 0,
                get_path(cwd, cable_length_error_file).c_str(), status);
    }
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_x_file))
    {
        oskar_station_load_cable_length_error(station, 0,
                get_path(cwd, cable_length_error_x_file).c_str(), status);
    }
    if (oskar_dir_file_exists(cwd.c_str(), cable_length_error_y_file))
    {
        oskar_station_load_cable_length_error(station, 1,
                get_path(cwd, cable_length_error_y_file).c_str(), status);
    }
}

string TelescopeLoaderCableLengthError::name() const
{
    return string("element cable length error file loader");
}
