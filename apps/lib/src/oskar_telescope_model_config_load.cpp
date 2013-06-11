/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "apps/lib/oskar_telescope_model_config_load.h"
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_station_model_copy.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_init_child_stations.h"
#include "station/oskar_station_model_resize.h"
#include "station/oskar_station_model_resize_element_types.h"
#include "station/oskar_station_model_load_config.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_station_model_set_element_orientation.h"
#include "station/oskar_station_model_set_element_weight.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_get_error_string.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QHash>

#include <cstdlib>

static const char config_file[] = "config.txt";
static const char layout_file[] = "layout.txt";

// Private function prototypes
//==============================================================================
static void load_directories(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings,
        const QDir& cwd, oskar_StationModel* station, int depth, int* status);

static void load_interferometer_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir, int* status);

static void load_station_config(oskar_StationModel* station, const QDir& dir,
        int* status);
//==============================================================================


extern "C"
void oskar_telescope_model_config_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings, int* status)
{
    // Check all inputs.
    if (!telescope || !settings || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    // Check if safe to proceed.
    if (*status) return;

    // Check that the directory exists.
    QDir telescope_dir(settings->input_directory);
    if (!telescope_dir.exists())
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    // Load layout.txt and config.txt files from the telescope directory tree.
    load_directories(telescope, settings, telescope_dir, NULL, 0, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to load telescope model (%s).",
                oskar_get_error_string(*status));
    }
}

// Private functions.

static void load_directories(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& cwd,
        oskar_StationModel* station, int depth, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Get a list of all (child) stations in this directory, sorted by name.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_dirs = children.size();

    // Load the interferometer layout if we're at depth 0 (top level directory).
    if (depth == 0)
    {
        load_interferometer_layout(telescope, settings, cwd, status);

        // Check the number of station directories found at the top level.
        if (num_dirs == 0)
        {
            // No directories. Set all "stations" to be a single dipole.
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                oskar_StationModel* station = &telescope->station[i];
                oskar_station_model_resize(station, 1, status);
                oskar_station_model_resize_element_types(station, 1, status);
                oskar_station_model_set_element_coords(station, 0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
                oskar_station_model_set_element_errors(station, 0,
                        1.0, 0.0, 0.0, 0.0, status);
                oskar_station_model_set_element_orientation(station, 0,
                        90.0, 0.0, status);
                oskar_station_model_set_element_weight(station, 0,
                        1.0, 0.0, status);
            }
        }
        else if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            QDir child_dir(cwd.filePath(children[0]));

            // Recursive call to load the station.
            load_directories(telescope, settings, child_dir,
                    &telescope->station[0], depth + 1, status);

            // Copy station 0 to all the others.
            for (int i = 1; i < telescope->num_stations; ++i)
            {
                oskar_station_model_copy(&telescope->station[i],
                        &telescope->station[0], status);
            }
        }
        else
        {
            // Consistency check.
            if (num_dirs != telescope->num_stations)
            {
                *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                return;
            }

            // Loop over and descend into all child stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Get the child directory.
                QDir child_dir(cwd.filePath(children[i]));

                // Recursive call to load this (child) station.
                load_directories(telescope, settings, child_dir,
                        &telescope->station[i], depth + 1, status);
            }
        } // End check on number of directories.
    }
    // At some other depth in the directory tree, load the station "config.txt".
    else
    {
        load_station_config(station, cwd, status);

        // Check if this is the last level.
        if (num_dirs == 0)
        {
            // Allocate storage for element model data if no children.
            oskar_station_model_resize_element_types(station, 1, status);
        }
        else
        {
            // Allocate storage for child stations.
            oskar_station_model_init_child_stations(station, status);

            if (num_dirs == 1)
            {
                // One station directory. Load and copy it to all the others.
                QDir child_dir(cwd.filePath(children[0]));

                // Recursive call to load the station.
                load_directories(telescope, settings, child_dir,
                        &station->child[0], depth + 1, status);

                // Copy child station 0 to all the others.
                for (int i = 1; i < station->num_elements; ++i)
                {
                    oskar_station_model_copy(&station->child[i],
                            &station->child[0], status);
                }
            }
            else
            {
                // Consistency check.
                if (num_dirs != station->num_elements)
                {
                    *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                    return;
                }

                // Loop over and descend into all child stations.
                for (int i = 0; i < num_dirs; ++i)
                {
                    // Get the child directory.
                    QDir child_dir(cwd.filePath(children[i]));

                    // Recursive call to load this (child) station.
                    load_directories(telescope, settings, child_dir,
                            &station->child[i], depth + 1, status);
                }
            }
        }
    }
}

static void load_interferometer_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Check for presence of "layout.txt" then "config.txt" (in that order).
    const char* file = NULL;
    if (dir.exists(layout_file))
        file = layout_file;
    else if (dir.exists(config_file))
        file = config_file;
    else
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;

    // Load the station positions.
    oskar_telescope_model_load_station_coords(telescope,
            dir.filePath(file).toLatin1(), settings->longitude_rad,
            settings->latitude_rad, settings->altitude_m, status);
}

static void load_station_config(oskar_StationModel* station, const QDir& dir,
        int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Check for presence of "config.txt".
    if (dir.exists(config_file))
    {
        oskar_station_model_load_config(station,
                dir.filePath(config_file).toLatin1(), status);
    }
    else
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;
}
