/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_get_num_procs.h"
#include "utility/oskar_thread.h"
#include "telescope/private_TelescopeLoaderApodisation.h"
#include "telescope/private_TelescopeLoaderCableLengthError.h"
#include "telescope/private_TelescopeLoaderElementPattern.h"
#include "telescope/private_TelescopeLoaderElementTypes.h"
#include "telescope/private_TelescopeLoaderFeedAngle.h"
#include "telescope/private_TelescopeLoaderGainModel.h"
#include "telescope/private_TelescopeLoaderGainPhase.h"
#include "telescope/private_TelescopeLoaderHarpData.h"
#include "telescope/private_TelescopeLoaderLayout.h"
#include "telescope/private_TelescopeLoaderMountTypes.h"
#include "telescope/private_TelescopeLoaderNoise.h"
#include "telescope/private_TelescopeLoaderPermittedBeams.h"
#include "telescope/private_TelescopeLoaderPosition.h"
#include "telescope/private_TelescopeLoaderStationTypeMap.h"

#include <cstdlib>
#include <map>
#include <string>
#include <vector>

using std::map;
using std::string;
using std::vector;

static void load_directories(oskar_Telescope* telescope,
        const string& cwd, oskar_Station* station, int depth,
        const vector<oskar_TelescopeLoadAbstract*>& loaders,
        map<string, string> filemap, oskar_Log* log, int* status);

extern "C"
void oskar_telescope_load(oskar_Telescope* telescope, const char* path,
        oskar_Log* log, int* status)
{
    if (*status) return;

    // Check that the telescope directory has been set and exists.
    if (!path || !oskar_dir_exists(path))
    {
        oskar_log_error(log,
                "Telescope model directory '%s' does not exist.", path);
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
    // The position loader must be first, because it defines the
    // reference coordinates. The layout loader must be next, because
    // it defines the stations.
    loaders.push_back(new TelescopeLoaderPosition);
    loaders.push_back(new TelescopeLoaderLayout);
    loaders.push_back(new TelescopeLoaderStationTypeMap);
    loaders.push_back(new TelescopeLoaderGainPhase);
    loaders.push_back(new TelescopeLoaderGainModel);
    loaders.push_back(new TelescopeLoaderHarpData);
    loaders.push_back(new TelescopeLoaderCableLengthError);
    loaders.push_back(new TelescopeLoaderApodisation);
    loaders.push_back(new TelescopeLoaderFeedAngle);
    loaders.push_back(new TelescopeLoaderElementTypes);
    loaders.push_back(new TelescopeLoaderMountTypes);
    loaders.push_back(new TelescopeLoaderPermittedBeams);
    loaders.push_back(new TelescopeLoaderElementPattern);
    loaders.push_back(new TelescopeLoaderNoise);

    // Load everything recursively from the telescope directory tree.
    map<string, string> filemap;
    load_directories(telescope, string(path), NULL, 0, loaders,
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

    // Set unique station IDs.
    oskar_telescope_set_station_ids_and_coords(telescope, status);
}

// Private functions.

struct oskar_ThreadArgs
{
    oskar_Telescope* telescope;
    const string* cwd;
    const vector<oskar_TelescopeLoadAbstract*>* loaders;
    map<string, string>* filemap;
    oskar_Log* log;
    int *status, i_thread, num_threads, num_dirs;
    const char* const* children;
    oskar_ThreadArgs(oskar_Telescope* telescope,
            const string& cwd,
            const vector<oskar_TelescopeLoadAbstract*>& loaders,
            map<string, string>& filemap, oskar_Log* log, int* status,
            int i_thread, int num_threads,
            int num_dirs, const char* const* children)
    : telescope(telescope), cwd(&cwd), loaders(&loaders), filemap(&filemap),
      log(log), status(status), i_thread(i_thread), num_threads(num_threads),
      num_dirs(num_dirs), children(children) {}
};
typedef struct oskar_ThreadArgs oskar_ThreadArgs;

static void* thread_func(void* arg)
{
    oskar_ThreadArgs* a = (oskar_ThreadArgs*) arg;
    for (int i = a->i_thread; i < a->num_dirs; i += a->num_threads)
    {
        load_directories(a->telescope,
                oskar_TelescopeLoadAbstract::get_path(*(a->cwd),
                a->children[i]), oskar_telescope_station(a->telescope, i), 1,
                *(a->loaders), *(a->filemap), a->log, a->status);
    }
    return 0;
}

// Must pass filemap by value rather than by reference; otherwise, recursive
// behaviour will not work as intended.
static void load_directories(oskar_Telescope* telescope,
        const string& cwd, oskar_Station* station, int depth,
        const vector<oskar_TelescopeLoadAbstract*>& loaders,
        map<string, string> filemap, oskar_Log* log, int* status)
{
    int num_dirs = 0;
    char** children = 0;
    if (*status) return;

    // Get a list of all (child) stations in this directory, sorted by name.
    oskar_dir_items(cwd.c_str(), NULL, 0, 1, &num_dirs, &children);

    // Top-level depth.
    if (depth == 0)
    {
        // Set the number of station models to be the number of directories.
        // Do this first!
        if (num_dirs == 0)
        {
            oskar_telescope_resize_station_array(telescope, 1, status);
        }
        else
        {
            oskar_telescope_resize_station_array(telescope, num_dirs, status);
        }

        // Load everything at this level.
        for (size_t i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(telescope, cwd, num_dirs, filemap, status);
            if (*status)
            {
                string s = string("Error in ") + loaders[i]->name() +
                        string(" in '") + cwd + string("'.");
                oskar_log_error(log, "%s", s.c_str());
                goto fail;
            }
        }

        if (num_dirs == 1)
        {
            // One station directory.
            // (Station type map will all be 0 by default in any case.)
            // Recursive call to load the station.
            load_directories(telescope,
                    oskar_TelescopeLoadAbstract::get_path(cwd, children[0]),
                    oskar_telescope_station(telescope, 0), depth + 1,
                    loaders, filemap, log, status);
        }
        else if (num_dirs > 1)
        {
            const int num_procs = oskar_get_num_procs();
            vector<oskar_Thread*> threads(num_procs);
            vector<oskar_ThreadArgs> args;

            // Check if "station_type_map.txt" exists.
            if (!oskar_dir_file_exists(cwd.c_str(), "station_type_map.txt"))
            {
                // Consistency check.
                if (num_dirs != oskar_telescope_num_stations(telescope))
                {
                    oskar_log_error(log,
                            "Inconsistent number of station model directories, "
                            "and no station type map found.");
                    *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                    goto fail;
                }
                else
                {
                    // Set implicit station type mapping.
                    oskar_telescope_set_unique_stations(telescope, 1, status);
                }
            }

            // Use multi-threading for load at top level only.
            for (int i = 0; i < num_procs; ++i)
            {
                args.push_back(oskar_ThreadArgs(telescope, cwd, loaders,
                        filemap, log, status, i, num_procs, num_dirs,
                        children));
            }
            for (int i = 0; i < num_procs; ++i)
            {
                threads[i] = oskar_thread_create(
                        thread_func, (void*)&args[i], 0);
            }
            for (int i = 0; i < num_procs; ++i)
            {
                oskar_thread_join(threads[i]);
                oskar_thread_free(threads[i]);
            }
        } // End check on number of directories.
    }

    // At some other depth.
    else if (station)
    {
        // Load everything at this level.
        for (size_t i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(station, cwd, num_dirs, depth, filemap, status);
            if (*status)
            {
                string s = string("Error in ") + loaders[i]->name() +
                        string(" in '") + cwd + string("'.");
                oskar_log_error(log, "%s", s.c_str());
                goto fail;
            }
        }

        if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            // Recursive call to load the station.
            load_directories(telescope,
                    oskar_TelescopeLoadAbstract::get_path(cwd, children[0]),
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
                goto fail;
            }

            // Loop over and descend into all stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Recursive call to load the station.
                load_directories(telescope,
                        oskar_TelescopeLoadAbstract::get_path(cwd, children[i]),
                        oskar_station_child(station, i), depth + 1, loaders,
                        filemap, log, status);
            }
        } // End check on number of directories.
    } // End check on depth.

fail:
    for (int i = 0; i < num_dirs; ++i) free(children[i]);
    free(children);
}
