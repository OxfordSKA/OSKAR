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

#include "apps/lib/oskar_dir.h"
#include "apps/lib/oskar_telescope_load.h"
#include "apps/lib/private_TelescopeLoadApodisation.h"
#include "apps/lib/private_TelescopeLoadElementTypes.h"
#include "apps/lib/private_TelescopeLoadElementPattern.h"
#include "apps/lib/private_TelescopeLoadFeedAngle.h"
#include "apps/lib/private_TelescopeLoadGainPhase.h"
#include "apps/lib/private_TelescopeLoadLayout.h"
#include "apps/lib/private_TelescopeLoadMountTypes.h"
#include "apps/lib/private_TelescopeLoadNoise.h"
#include "apps/lib/private_TelescopeLoadPermittedBeams.h"
#include <oskar_log.h>
#include <oskar_get_error_string.h>

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::vector;

// Private function prototype.
static void load_directories(oskar_Telescope* telescope,
        const oskar_Settings* settings, const oskar_Dir& cwd,
        oskar_Station* station, int depth,
        const vector<oskar_TelescopeLoadAbstract*>& loaders,
        map<string, string> filemap, oskar_Log* log, int* status);


extern "C"
void oskar_telescope_load(oskar_Telescope* telescope, oskar_Log* log,
        const oskar_Settings* settings, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Check that the telescope directory has been set.
    if (!settings->telescope.input_directory)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    // Check that the directory exists.
    oskar_Dir telescope_dir(string(settings->telescope.input_directory));
    if (!telescope_dir.exists())
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_mem_location(telescope) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    // Create the loaders.
    vector<oskar_TelescopeLoadAbstract*> loaders;
    // The layout loader must be first.
    loaders.push_back(new TelescopeLoadLayout(settings));
    loaders.push_back(new TelescopeLoadGainPhase);
    loaders.push_back(new TelescopeLoadApodisation);
    loaders.push_back(new TelescopeLoadFeedAngle);
    loaders.push_back(new TelescopeLoadElementTypes);
    loaders.push_back(new TelescopeLoadMountTypes);
    loaders.push_back(new TelescopeLoadPermittedBeams);
    loaders.push_back(new TelescopeLoadElementPattern(settings, log));
    loaders.push_back(new TelescopeLoadNoise(settings));

    // Load everything recursively from the telescope directory tree.
    map<string, string> filemap;
    load_directories(telescope, settings, telescope_dir, NULL, 0, loaders,
            filemap, log, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to load telescope model (%s).",
                oskar_get_error_string(*status));
    }

    // Delete all the loaders.
    for (size_t i = 0; i < loaders.size(); ++i)
    {
        delete loaders[i];
    }
}

// Private functions.

// Must pass filemap by value rather than by reference; otherwise, recursive
// behaviour will not work as intended.
static void load_directories(oskar_Telescope* telescope,
        const oskar_Settings* settings, const oskar_Dir& cwd,
        oskar_Station* station, int depth,
        const vector<oskar_TelescopeLoadAbstract*>& loaders,
        map<string, string> filemap, oskar_Log* log, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Get a list of all (child) stations in this directory, sorted by name.
    vector<string> children = cwd.allSubDirs();
    int num_dirs = children.size();

    // Top-level depth.
    if (depth == 0)
    {
        // Load everything at this level.
        for (size_t i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(telescope, cwd, num_dirs, filemap, status);
            if (*status)
            {
                string s = string("Error in ") + loaders[i]->name() +
                        string(" in '") + cwd.absolutePath() + string("'.");
                oskar_log_error(log, "%s", s.c_str());
                return;
            }
        }

        if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            oskar_Dir child_dir(cwd.filePath(children[0]));

            // Recursive call to load the station.
            load_directories(telescope, settings, child_dir,
                    oskar_telescope_station(telescope, 0), depth + 1,
                    loaders, filemap, log, status);

            // Copy station 0 to all the others.
            oskar_telescope_duplicate_first_station(telescope, status);
        }
        else if (num_dirs > 1)
        {
            // Consistency check.
            if (num_dirs != oskar_telescope_num_stations(telescope))
            {
                *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                return;
            }

            // Loop over and descend into all stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Get the child directory.
                oskar_Dir child_dir(cwd.filePath(children[i]));

                // Recursive call to load the station.
                load_directories(telescope, settings, child_dir,
                        oskar_telescope_station(telescope, i), depth + 1,
                        loaders, filemap, log, status);
            }
        } // End check on number of directories.
    }

    // At some other depth.
    else
    {
        // Load everything at this level.
        for (size_t i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(station, cwd, num_dirs, depth, filemap, status);
            if (*status)
            {
                string s = string("Error in ") + loaders[i]->name() +
                        string(" in '") + cwd.absolutePath() + string("'.");
                oskar_log_error(log, "%s", s.c_str());
                return;
            }
        }

        if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            oskar_Dir child_dir(cwd.filePath(children[0]));

            // Recursive call to load the station.
            load_directories(telescope, settings, child_dir,
                    oskar_station_child(station, 0), depth + 1, loaders,
                    filemap, log, status);

            // Copy station 0 to all the others.
            oskar_station_duplicate_first_child(station, status);
        }
        else if (num_dirs > 1)
        {
            // Consistency check.
            if (num_dirs != oskar_station_num_elements(station))
            {
                *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                return;
            }

            // Loop over and descend into all stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Get the child directory.
                oskar_Dir child_dir(cwd.filePath(children[i]));

                // Recursive call to load the station.
                load_directories(telescope, settings, child_dir,
                        oskar_station_child(station, i), depth + 1, loaders,
                        filemap, log, status);
            }
        } // End check on number of directories.
    } // End check on depth.
}
