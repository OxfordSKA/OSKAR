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

#include "apps/lib/oskar_telescope_model_config_load.h"
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_load_config.h"
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

static void load_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_stations, int* status);

static void load_config(oskar_StationModel* station, const QDir& dir,
        int* status);

static void allocate_children(oskar_StationModel* station,
        int num_station_dirs, int type, int* status);
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

// Private functions

static void load_directories(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& cwd,
        oskar_StationModel* station, int depth, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Get a list of all (child) stations in this directory, sorted by name.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_children = children.count();

    // Load the interferometer layout if we're at depth 0 (top level directory).
    if (depth == 0)
    {
        load_layout(telescope, settings, cwd, num_children, status);
    }
    // At some other depth in the directory tree, load the station config.txt
    else
    {
        load_config(station, cwd, status);

        // If any children exist, allocate storage for them in the model.
        if (num_children > 0)
        {
            int type = oskar_telescope_model_type(telescope);
            allocate_children(station, num_children, type, status);
        }
    }
    if (*status) return;

    // Loop over and descend into all child stations.
    for (int i = 0; i < num_children; ++i)
    {
        // Get a pointer to the child station.
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Get the child directory.
        QDir child_dir(cwd.filePath(children[i]));

        // Load this (child) station.
        load_directories(telescope, settings, child_dir, s, depth + 1, status);
    }
}

static void load_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_stations, int* status)
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
    {
        *status = OSKAR_ERR_SETUP_FAIL;
        return;
    }

    // Get the full path to the file.
    QByteArray path = dir.filePath(file).toAscii();

    // Load the station positions.
    oskar_telescope_model_load_station_coords(telescope,
            path, settings->longitude_rad, settings->latitude_rad,
            settings->altitude_m, status);
    if (*status) return;

    // Check that there are the right number of stations.
    if (num_stations > 0)
    {
        if (num_stations != telescope->num_stations)
            *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
    }
    else
    {
        // TODO There are no station directories.
        // Still need to set up the stations, though.
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE;
    }
}

static void load_config(oskar_StationModel* station, const QDir& dir,
        int* status)
{
    // Check for presence of "config.txt".
    if (dir.exists(config_file))
    {
        QByteArray path = dir.filePath(config_file).toAscii();
        oskar_station_model_load_config(station, path, status);
    }
    else
        *status = OSKAR_ERR_SETUP_FAIL;
}

static void allocate_children(oskar_StationModel* station, int num_children,
        int type, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Check that there are the right number of stations.
    if (num_children != station->num_elements)
    {
        *status = OSKAR_ERR_SETUP_FAIL;
        return;
    }

    // Allocate memory for child station array.
    station->child = (oskar_StationModel*) malloc(num_children *
            sizeof(oskar_StationModel));

    // Initialise each child station.
    for (int i = 0; i < num_children; ++i)
    {
        oskar_station_model_init(&station->child[i], type,
                OSKAR_LOCATION_CPU, 0, status);
    }
}
