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

#include "apps/lib/oskar_telescope_model_element_pattern_load.h"

#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_telescope_model_resize.h"
#include "station/oskar_ElementModel.h"
#include "station/oskar_element_model_copy.h"
#include "station/oskar_element_model_init.h"
#include "station/oskar_element_model_load_cst.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_load_config.h"
#include "station/oskar_station_model_resize_element_types.h"
#include "utility/oskar_log_message.h"
#include "utility/oskar_log_error.h"
#include "utility/oskar_get_error_string.h"

#include <cstdlib>
#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QHash>

// Element pattern filenames.
static const char element_x_cst_file[] = "element_pattern_x_cst.txt";
static const char element_y_cst_file[] = "element_pattern_y_cst.txt";

// Private functions.
static void load_directories(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const QDir& cwd, oskar_StationModel* station,
        int depth, QHash<QString, QString>& files,
        QHash<QString, oskar_ElementModel*>& models, int* status);
static void load_element_patterns(oskar_Log* log,
        const oskar_SettingsTelescope* settings,
        oskar_StationModel* station, QHash<QString, QString>& data_files,
        QHash<QString, oskar_ElementModel*>& models, int* status);

extern "C"
void oskar_telescope_model_element_pattern_load(oskar_TelescopeModel* telescope,
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

    if (!settings->aperture_array.element_pattern.enable_numerical_patterns)
        return;

    QHash<QString, oskar_ElementModel*> models;
    QHash<QString, QString> data_files;

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

    // Load element data by scanning the directory structure and loading
    // the element files deepest in the tree.
    load_directories(telescope, log, settings, telescope_dir, NULL,
            0, data_files, models, status);
    if (*status)
        oskar_log_error(log, "Loading element pattern files (%s).",
                oskar_get_error_string(*status));
}


// Private functions

static void load_directories(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const QDir& cwd, oskar_StationModel* station,
        int depth, QHash<QString, QString>& files,
        QHash<QString, oskar_ElementModel*>& models, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    // Update the dictionary of element files for the current station directory.
    if (cwd.exists(element_x_cst_file))
        files[QString(element_x_cst_file)] = cwd.absoluteFilePath(element_x_cst_file);
    if (cwd.exists(element_y_cst_file))
        files[QString(element_y_cst_file)] = cwd.absoluteFilePath(element_y_cst_file);

    // Get a list of the child stations.
    QStringList children;
    children = cwd.entryList(QDir::AllDirs | QDir::NoDotAndDotDot, QDir::Name);
    int num_dirs = children.size();

    // If the station / child arrays haven't been allocated
    // (by oskar_telescope_load_config() for example), allocate them.
    if (depth == 0 && telescope->station == NULL)
    {
        oskar_telescope_model_resize(telescope, num_dirs, status);
    }
    else if (depth > 0 && num_dirs > 0 && station->child == NULL)
    {
        int type = oskar_telescope_model_type(telescope);
        station->child = (oskar_StationModel*) malloc(num_dirs *
                sizeof(oskar_StationModel));
        if (!station->child)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }
        for (int i = 0; i < num_dirs; ++i)
        {
            oskar_station_model_init(&station->child[i], type,
                    OSKAR_LOCATION_CPU, 0, status);
        }
    }

    // Loop over, and descend into the child stations.
    for (int i = 0; i < num_dirs; ++i)
    {
        // Get a pointer to the child station.
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Get the child directory.
        QDir child_dir(cwd.filePath(children[i]));

        // Load this (child) station.
        load_directories(telescope, log, settings, child_dir, s,
                depth + 1, files, models, status);
    }

    // If there are no children, load the element pattern data corresponding
    // to the deepest element file found in the tree.
    if (num_dirs == 0)
    {
        if (station)
        {
            oskar_station_model_resize_element_types(station, 1, status);
            load_element_patterns(log, settings, station, files, models, status);
        }
        else
        {
            for (int i = 0; i < telescope->num_stations; ++i)
            {
                oskar_StationModel* s = &telescope->station[i];
                oskar_station_model_resize_element_types(s, 1, status);
                load_element_patterns(log, settings, s, files, models, status);
            }
        }
    }
}


static void load_element_patterns(oskar_Log* log,
        const oskar_SettingsTelescope* settings, oskar_StationModel* station,
        QHash<QString, QString>& data_files,
        QHash<QString, oskar_ElementModel*>& models, int* status)
{
    // Check if safe to proceed.
    if (*status) return;

    QByteArray element_x, element_y;
    const char *element_file_x = NULL, *element_file_y = NULL;

    if (data_files.contains(QString(element_x_cst_file)))
    {
        element_x = data_files.value(QString(element_x_cst_file)).toAscii();
        element_file_x = element_x.constData();
    }
    if (data_files.contains(QString(element_y_cst_file)))
    {
        element_y = data_files.value(QString(element_y_cst_file)).toAscii();
        element_file_y = element_y.constData();
    }

    QString files;
    if (element_file_x) files.append(element_file_x);
    if (element_file_y) files.append(element_file_y);
    if (files.length() > 0)
    {
        // Check if this file combination has already been loaded.
        if (models.contains(files))
        {
            // Copy the element pattern data.
            oskar_element_model_copy(station->element_pattern,
                    models.value(files), status);
        }
        else
        {
            // Load CST element pattern data.
            if (element_file_x)
            {
                oskar_log_message(log, 0, "Loading CST element "
                        "pattern data (X): %s", element_file_x);
                oskar_log_message(log, 0, "");
                oskar_element_model_load_cst(station->element_pattern,
                        log, 1, element_file_x,
                        &settings->aperture_array.element_pattern.fit, status);
            }
            if (element_file_y)
            {
                oskar_log_message(log, 0, "Loading CST element "
                        "pattern data (Y): %s", element_file_y);
                oskar_log_message(log, 0, "");
                oskar_element_model_load_cst(station->element_pattern,
                        log, 2, element_file_y,
                        &settings->aperture_array.element_pattern.fit, status);
            }

            // Store pointer to the element model for these files.
            models.insert(files, station->element_pattern);
        }
    }
}
