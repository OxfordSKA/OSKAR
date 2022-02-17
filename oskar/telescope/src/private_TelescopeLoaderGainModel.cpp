/*
 * Copyright (c) 2020-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderGainModel.h"
#include "utility/oskar_dir.h"

using std::map;
using std::string;

static const char* gain_model_file = "gain_model.h5";

void TelescopeLoaderGainModel::load(oskar_Telescope* telescope,
        const string& cwd, int /*num_subdirs*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of gain model data.
    if (oskar_dir_file_exists(cwd.c_str(), gain_model_file))
    {
        oskar_gains_open_hdf5(oskar_telescope_gains(telescope),
                get_path(cwd, gain_model_file).c_str(), status);
    }
}

void TelescopeLoaderGainModel::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int /*depth*/,
        map<string, string>& /*filemap*/, int* status)
{
    // Check for presence of gain model data.
    if (oskar_dir_file_exists(cwd.c_str(), gain_model_file))
    {
        oskar_gains_open_hdf5(oskar_station_gains(station),
                get_path(cwd, gain_model_file).c_str(), status);
    }
}

string TelescopeLoaderGainModel::name() const
{
    return string("gain model loader");
}
