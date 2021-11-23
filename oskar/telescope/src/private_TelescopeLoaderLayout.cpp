/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderLayout.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* layout_file = "layout.txt";
static const char* layout_enu_file = "layout_enu.txt";
static const char* layout_x_file = "layout_x.txt";
static const char* layout_enu_x_file = "layout_enu_x.txt";
static const char* layout_y_file = "layout_y.txt";
static const char* layout_enu_y_file = "layout_enu_y.txt";
static const char* layout_ecef_file = "layout_ecef.txt";
static const char* layout_wgs84_file = "layout_wgs84.txt";


void TelescopeLoaderLayout::load(oskar_Telescope* telescope,
        const string& cwd, int num_subdirs,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of top-level layout files.
    if (oskar_dir_file_exists(cwd.c_str(), layout_file))
    {
        // Load the interferometer layout (horizon plane).
        oskar_telescope_load_station_coords_enu(telescope,
                get_path(cwd, layout_file).c_str(),
                oskar_telescope_lon_rad(telescope),
                oskar_telescope_lat_rad(telescope),
                oskar_telescope_alt_metres(telescope), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_enu_file))
    {
        // Load the interferometer layout (horizon plane).
        oskar_telescope_load_station_coords_enu(telescope,
                get_path(cwd, layout_enu_file).c_str(),
                oskar_telescope_lon_rad(telescope),
                oskar_telescope_lat_rad(telescope),
                oskar_telescope_alt_metres(telescope), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_ecef_file))
    {
        // Load the interferometer layout (ECEF system).
        oskar_telescope_load_station_coords_ecef(telescope,
                get_path(cwd, layout_ecef_file).c_str(),
                oskar_telescope_lon_rad(telescope),
                oskar_telescope_lat_rad(telescope),
                oskar_telescope_alt_metres(telescope), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_wgs84_file))
    {
        // Load the interferometer layout (WGS84 system).
        oskar_telescope_load_station_coords_wgs84(telescope,
                get_path(cwd, layout_wgs84_file).c_str(),
                oskar_telescope_lon_rad(telescope),
                oskar_telescope_lat_rad(telescope),
                oskar_telescope_alt_metres(telescope), status);
    }
    else
    {
        // If telescope hasn't already been sized, return an error.
        if (oskar_telescope_num_stations(telescope) == 0)
        {
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;
        }
        return;
    }

    // If no subdirectories, set all "stations" to be a single dipole.
    if (num_subdirs == 0)
    {
        int num_stations = oskar_telescope_num_station_models(telescope);
        for (int i = 0; i < num_stations; ++i)
        {
            oskar_Station* station = oskar_telescope_station(telescope, i);
            oskar_station_resize(station, 1, status);
            oskar_station_resize_element_types(station, 1, status);
        }
    }
}

void TelescopeLoaderLayout::load(oskar_Station* station,
        const string& cwd, int num_subdirs, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of station layout file.
    if (oskar_dir_file_exists(cwd.c_str(), layout_file))
    {
        oskar_station_load_layout(station, 0,
                get_path(cwd, layout_file).c_str(), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_enu_file))
    {
        oskar_station_load_layout(station, 0,
                get_path(cwd, layout_enu_file).c_str(), status);
    }

    if (oskar_dir_file_exists(cwd.c_str(), layout_x_file))
    {
        oskar_station_load_layout(station, 0,
                get_path(cwd, layout_x_file).c_str(), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_enu_x_file))
    {
        oskar_station_load_layout(station, 0,
                get_path(cwd, layout_enu_x_file).c_str(), status);
    }

    if (oskar_dir_file_exists(cwd.c_str(), layout_y_file))
    {
        oskar_station_load_layout(station, 1,
                get_path(cwd, layout_y_file).c_str(), status);
    }
    else if (oskar_dir_file_exists(cwd.c_str(), layout_enu_y_file))
    {
        oskar_station_load_layout(station, 1,
                get_path(cwd, layout_enu_y_file).c_str(), status);
    }

    // If station hasn't already been sized, return an error.
    if (oskar_station_num_elements(station) == 0)
    {
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;
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

string TelescopeLoaderLayout::name() const
{
    return string("layout file loader");
}
