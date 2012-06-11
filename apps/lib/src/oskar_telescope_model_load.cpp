/*
 * Copyright (c) 2012, The University of Oxford
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
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "station/oskar_ElementModel.h"
#include "station/oskar_element_model_copy.h"
#include "station/oskar_element_model_init.h"
#include "station/oskar_element_model_load_cst.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_station_model_load_config.h"
#include "utility/oskar_log_message.h"

#include <cstdlib>
#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QHash>

static const char config_name[] = "config.txt";
static const char layout_name[] = "layout.txt";
static const char element_x_name_cst[] = "element_pattern_x_cst.txt";
static const char element_y_name_cst[] = "element_pattern_y_cst.txt";

static int oskar_telescope_model_load_private(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const char* dir_path, oskar_StationModel* station,
        const char* element_file_x, const char* element_file_y, int depth,
        QHash<QString, oskar_ElementModel*>& models);

extern "C"
int oskar_telescope_model_load(oskar_TelescopeModel* telescope, oskar_Log* log,
        const oskar_SettingsTelescope* settings)
{
    QHash<QString, oskar_ElementModel*> models;
    return oskar_telescope_model_load_private(telescope, log, settings,
            settings->config_directory, NULL, NULL, NULL, 0, models);
}

static int oskar_telescope_model_load_private(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const char* dir_path, oskar_StationModel* station,
        const char* element_file_x, const char* element_file_y, int depth,
        QHash<QString, oskar_ElementModel*>& models)
{
    int error;

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the directory exists.
    QDir dir;
    dir.setPath(dir_path);
    if (!dir.exists())
        return OSKAR_ERR_FILE_IO;

    // Get a list of all stations in this directory, sorted by name.
    QStringList stations = dir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot,
            QDir::Name);
    int num_station_dirs = stations.size();

    // Check for CST element pattern data.
    QByteArray element_x, element_y;
    if (dir.exists(element_x_name_cst))
    {
        element_x = dir.absoluteFilePath(element_x_name_cst).toAscii();
        element_file_x = element_x.constData();
    }
    if (dir.exists(element_y_name_cst))
    {
        element_y = dir.absoluteFilePath(element_y_name_cst).toAscii();
        element_file_y = element_y.constData();
    }

    // Load the station layout file if we're at depth 0.
    if (depth == 0)
    {
        // Check for presence of "layout.txt" or "config.txt".
        const char* coord_file = NULL;
        if (dir.exists(layout_name))
            coord_file = layout_name; // Override.
        else if (dir.exists(config_name))
            coord_file = config_name;
        else
            return OSKAR_ERR_SETUP_FAIL;

        // Load the station positions.
        QByteArray coord_path = dir.filePath(coord_file).toAscii();
        error = oskar_telescope_model_load_station_coords(telescope,
                coord_path, settings->longitude_rad, settings->latitude_rad,
                settings->altitude_m);
        if (error) return error;

        // Check that there are the right number of stations.
        if (num_station_dirs > 0)
        {
            if (num_station_dirs != telescope->num_stations)
                return OSKAR_ERR_SETUP_FAIL;
        }
        else
        {
            // There are no station directories.
            // Still need to set up the stations, though.
            return OSKAR_ERR_SETUP_FAIL;
        }
    }
    else
    {
        // Check for presence of "config.txt".
        if (dir.exists(config_name))
        {
            // Load the station data.
            QByteArray config_path = dir.filePath(config_name).toAscii();
            error = oskar_station_model_load_config(station,
                    config_path);
            if (error) return error;
        }
        else
            return OSKAR_ERR_SETUP_FAIL;

        if (num_station_dirs > 0)
        {
            // Check that there are the right number of stations.
            if (num_station_dirs != station->num_elements)
                return OSKAR_ERR_SETUP_FAIL;

            // Allocate memory for child station array.
            station->child = (oskar_StationModel*) malloc(num_station_dirs *
                    sizeof(oskar_StationModel));

            // Initialise each child station.
            for (int i = 0; i < num_station_dirs; ++i)
            {
                error = oskar_station_model_init(&station->child[i],
                        oskar_telescope_model_type(telescope),
                        OSKAR_LOCATION_CPU, 0);
                if (error) return error;
            }
        }
        else
        {
            // There are no child stations.
            if (!settings->station.ignore_custom_element_patterns)
            {
                // Load element pattern data for the station,
                // if files have been found.
                if (element_file_x || element_file_y)
                {
                    // Allocate memory for the element pattern structure.
                    station->element_pattern = (oskar_ElementModel*)
                                malloc(sizeof(oskar_ElementModel));
                    error = oskar_element_model_init(station->element_pattern,
                            oskar_telescope_model_type(telescope),
                            OSKAR_LOCATION_CPU);
                    if (error) return error;
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
                        error = oskar_element_model_copy(
                                station->element_pattern, models.value(files));
                        if (error) return error;
                    }
                    else
                    {
                        // Load CST element pattern data.
                        if (element_file_x)
                        {
                            oskar_log_message(log, 0, "Loading CST element "
                                    "pattern data (X): %s", element_file_x);
                            oskar_log_message(log, 0, "");
                            error = oskar_element_model_load_cst(
                                    station->element_pattern, log, 1,
                                    element_file_x,
                                    &settings->station.element_fit);
                            if (error) return error;
                        }
                        if (element_file_y)
                        {
                            oskar_log_message(log, 0, "Loading CST element "
                                    "pattern data (Y): %s", element_file_y);
                            oskar_log_message(log, 0, "");
                            error = oskar_element_model_load_cst(
                                    station->element_pattern, log, 2,
                                    element_file_y,
                                    &settings->station.element_fit);
                            if (error) return error;
                        }

                        // Store pointer to the element model for these files.
                        models.insert(files, station->element_pattern);
                    }
                }
            }
        }
    }

    // Loop over all station directories.
    for (int i = 0; i < num_station_dirs; ++i)
    {
        // Get the name of the station, and a pointer to the station to fill.
        QByteArray station_name = dir.filePath(stations[i]).toAscii();
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Load this station.
        error = oskar_telescope_model_load_private(telescope, log, settings,
                station_name, s, element_file_x, element_file_y, depth + 1,
                models);
        if (error) return error;
    }

    return 0;
}
