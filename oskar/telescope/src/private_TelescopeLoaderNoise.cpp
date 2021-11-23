/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_TelescopeLoaderNoise.h"
#include "utility/oskar_dir.h"

#include <cfloat>
#include <cassert>
#include <cmath>

using std::map;
using std::string;

TelescopeLoaderNoise::TelescopeLoaderNoise()
: oskar_TelescopeLoadAbstract(), freqs_(0), telescope_(0)
{
    files_[FREQ] = "noise_frequencies.txt";
    files_[RMS]  = "rms.txt";
}

TelescopeLoaderNoise::~TelescopeLoaderNoise()
{
    int status = 0;
    oskar_mem_free(freqs_, &status);
}


// Depth = 0
// - Set up frequency data as this is the same for all stations
//   and if defined by files these have to be at depth 0.
void TelescopeLoaderNoise::load(oskar_Telescope* telescope, const string& cwd,
        int num_subdirs, map<string, string>& filemap, int* status)
{
    string filename;
    telescope_ = telescope;
    if (*status) return;

    // Update the noise files for the current station directory.
    update_map(filemap, cwd);

    // Load the frequency file if it exists (this only happens at depth = 0).
    if (filemap.count(files_[FREQ]))
    {
        filename = filemap.at(files_[FREQ]);
    }
    if (!filename.empty())
    {
        freqs_ = oskar_mem_create(oskar_telescope_precision(telescope),
                OSKAR_CPU, 0, status);
        oskar_mem_load_ascii(filename.c_str(), 1, status, freqs_, "");

        // If no sub-directories (the station load function is never called)
        if (num_subdirs == 0)
        {
            int num_stations = oskar_telescope_num_station_models(telescope);
            for (int i = 0; i < num_stations; ++i)
            {
                oskar_Station* s = oskar_telescope_station(telescope, i);
                if (!s) continue;
                oskar_mem_copy(oskar_station_noise_freq_hz(s), freqs_, status);
                set_noise_rms(s, filemap, status);
            }
        }
    }
}


// Depth > 0
void TelescopeLoaderNoise::load(oskar_Station* station,
        const string& cwd, int /*num_subdirs*/, int depth,
        map<string, string>& filemap, int* status)
{
    if (*status) return;

    // Ignore noise files defined deeper than at the station level (depth == 1)
    // - Currently, noise is implemented as a additive term per station
    //   into the visibilities so using files at any other depth would be
    //   meaningless.
    if (depth > 1) return;

    // Update the noise files for the current station directory.
    update_map(filemap, cwd);

    if (freqs_)
    {
        // Set the frequency noise data field of the station structure.
        oskar_mem_copy(oskar_station_noise_freq_hz(station), freqs_, status);

        // Set the noise RMS.
        set_noise_rms(station, filemap, status);
    }
}

string TelescopeLoaderNoise::name() const
{
    return string("noise loader");
}


void TelescopeLoaderNoise::update_map(map<string, string>& filemap,
        const string& cwd)
{
    for (map<FileIds_, string>::const_iterator it = files_.begin();
            it != files_.end(); ++it)
    {
        string file = it->second;
        if (oskar_dir_file_exists(cwd.c_str(), file.c_str()))
        {
            filemap[file] = get_path(cwd, file);
        }
    }
}


void TelescopeLoaderNoise::set_noise_rms(oskar_Station* station,
        const map<string, string>& filemap, int* status)
{
    string filename;
    if (*status) return;

    if (filemap.count(files_[RMS])) filename = filemap.at(files_[RMS]);
    if (!filename.empty())
    {
        oskar_mem_load_ascii(filename.c_str(), 1, status,
                oskar_station_noise_rms_jy(station), "");
    }
}
