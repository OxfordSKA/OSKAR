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

#include "apps/lib/oskar_telescope_model_load.h"
#include "apps/lib/oskar_AbstractTelescopeFileLoader.h"
#include "apps/lib/oskar_ConfigFileLoader.h"
#include "apps/lib/oskar_ElementPatternLoader.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "station/oskar_station_model_copy.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_get_error_string.h"

#include <QtCore/QDir>
#include <QtCore/QHash>
#include <QtCore/QList>
#include <QtCore/QStringList>

#include <cstdlib>

// Private function prototype.
static void load_directories(oskar_TelescopeModel* telescope,
        const oskar_Settings* settings, const QDir& cwd,
        oskar_StationModel* station, int depth,
        const QList<oskar_AbstractTelescopeFileLoader*>& loaders,
        QHash<QString, QString> filemap, int* status);


extern "C"
void oskar_telescope_model_load(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_Settings* settings, int* status)
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
    QDir telescope_dir(settings->telescope.input_directory);
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

    // Create the loaders.
    QList<oskar_AbstractTelescopeFileLoader*> loaders;
    loaders.push_back(new oskar_ConfigFileLoader(settings)); // Must be first!
    loaders.push_back(new oskar_ElementPatternLoader(settings, log));

    // Load everything recursively from the telescope directory tree.
    QHash<QString, QString> filemap;
    load_directories(telescope, settings, telescope_dir, NULL, 0, loaders,
            filemap, status);
    if (*status)
    {
        oskar_log_error(log, "Failed to load telescope model (%s).",
                oskar_get_error_string(*status));
    }

    // Delete all the loaders.
    for (int i = 0; i < loaders.size(); ++i)
    {
        delete loaders[i];
    }
}

// Private functions.

static void load_directories(oskar_TelescopeModel* telescope,
        const oskar_Settings* settings, const QDir& cwd,
        oskar_StationModel* station, int depth,
        const QList<oskar_AbstractTelescopeFileLoader*>& loaders,
        QHash<QString, QString> filemap, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Get a list of all (child) stations in this directory, sorted by name.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_dirs = children.size();

    // Top-level depth.
    if (depth == 0)
    {
        // Load everything at this level.
        for (int i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(telescope, cwd, num_dirs, filemap, status);
            if (*status) return;
        }

        if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            QDir child_dir(cwd.filePath(children[0]));

            // Recursive call to load the station.
            load_directories(telescope, settings, child_dir,
                    &telescope->station[0], depth + 1, loaders, filemap,
                    status);

            // Copy station 0 to all the others.
            for (int i = 1; i < telescope->num_stations; ++i)
            {
                oskar_station_model_copy(&telescope->station[i],
                        &telescope->station[0], status);
            }
        }
        else if (num_dirs > 1)
        {
            // Consistency check.
            if (num_dirs != telescope->num_stations)
            {
                *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                return;
            }

            // Loop over and descend into all stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Get the child directory.
                QDir child_dir(cwd.filePath(children[i]));

                // Recursive call to load the station.
                load_directories(telescope, settings, child_dir,
                        &telescope->station[i], depth + 1, loaders, filemap,
                        status);
            }
        } // End check on number of directories.
    }

    // At some other depth.
    else
    {
        // Load everything at this level.
        for (int i = 0; i < loaders.size(); ++i)
        {
            loaders[i]->load(station, cwd, num_dirs, depth, filemap, status);
            if (*status) return;
        }

        if (num_dirs == 1)
        {
            // One station directory. Load and copy it to all the others.
            QDir child_dir(cwd.filePath(children[0]));

            // Recursive call to load the station.
            load_directories(telescope, settings, child_dir,
                    &station->child[0], depth + 1, loaders, filemap, status);

            // Copy station 0 to all the others.
            for (int i = 1; i < station->num_elements; ++i)
            {
                oskar_station_model_copy(&station->child[i],
                        &station->child[0], status);
            }
        }
        else if (num_dirs > 1)
        {
            // Consistency check.
            if (num_dirs != station->num_elements)
            {
                *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_ENTRIES_MISMATCH;
                return;
            }

            // Loop over and descend into all stations.
            for (int i = 0; i < num_dirs; ++i)
            {
                // Get the child directory.
                QDir child_dir(cwd.filePath(children[i]));

                // Recursive call to load the station.
                load_directories(telescope, settings, child_dir,
                        &station->child[i], depth + 1, loaders, filemap,
                        status);
            }
        } // End check on number of directories.
    } // End check on depth.
}
