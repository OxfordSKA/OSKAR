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

// Private function prototypes
//==============================================================================
static int oskar_telescope_model_load_private(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const char* dir_path, oskar_StationModel* station,
        int depth, QHash<QString, QString> data_files,
        QHash<QString, oskar_ElementModel*>& element_models);

static void oskar_telescope_model_get_data_files(
        QHash<QString, QString>& data_files, const QDir& dir, int depth);

static int oskar_telescope_model_load_layout(oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_station_dirs);

static int oskar_telescope_model_load_station_config(oskar_StationModel* station,
        const QDir& dir);

static int oskar_telescope_model_allocate_children(oskar_StationModel* station,
        int num_station_dirs, int type);

static int oskar_telescope_model_load_element_patterns(
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        oskar_StationModel* station, QHash<QString, QString> data_files,
        QHash<QString, oskar_ElementModel*>& models);

//==============================================================================

extern "C"
int oskar_telescope_model_load(oskar_TelescopeModel* telescope, oskar_Log* log,
        const oskar_SettingsTelescope* settings)
{
    QHash<QString, oskar_ElementModel*> models;
    QHash<QString, QString> data_files;
    return oskar_telescope_model_load_private(telescope, log, settings,
            settings->config_directory, NULL, 0, data_files, models);
}




// Private functions

static int oskar_telescope_model_load_private(oskar_TelescopeModel* telescope,
        oskar_Log* log, const oskar_SettingsTelescope* settings,
        const char* dir_path, oskar_StationModel* station,
        int depth, QHash<QString, QString> data_files,
        QHash<QString, oskar_ElementModel*>& element_models)
{
    int error;

    // Check that the telescope model is in CPU memory.
    if (oskar_telescope_model_location(telescope) != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    // Check that the directory exists.
    QDir dir(dir_path);
    if (!dir.exists()) return OSKAR_ERR_FILE_IO;

    // Get the set of data files in the current directory
    oskar_telescope_model_get_data_files(data_files, dir, depth);

    // Get a list of all stations in this directory, sorted by name.
    QStringList stations = dir.entryList(QDir::AllDirs | QDir::NoDotAndDotDot,
            QDir::Name);
    int num_station_dirs = stations.size();

    // Load the station layout file if we're at depth 0.
    if (depth == 0)
    {
        // Loads the 'config.txt' or 'layout.txt' at depth == 0.
        error = oskar_telescope_model_load_layout(telescope, settings, dir, num_station_dirs);
        if (error) return error;
    }
    else // at some other depth in the directory tree.
    {
        // Loads 'config.txt' for level > 0.
        error = oskar_telescope_model_load_station_config(station, dir);
        if (error) return error;

        // Check if any child stations exist. (note: at depth 0 the child
        // stations are allocated when loading layout)
        if (num_station_dirs > 0)
        {
            // Allocate and initialise child stations.
            error = oskar_telescope_model_allocate_children(station,
                    num_station_dirs, oskar_telescope_model_type(telescope));
            if (error) return error;
        }
        else // No child stations -> we are at the max depth in the tree.
        {
            // Load element pattern data
            if (!settings->station.ignore_custom_element_patterns)
            {
                error = oskar_telescope_model_load_element_patterns(log, settings,
                        station, data_files, element_models);
                if (error) return error;
            }
        }
    }

    // Loop over all station (child) directories.
    for (int i = 0; i < num_station_dirs; ++i)
    {
        // Get the name of the station, and a pointer to the station to fill.
        QByteArray station_name = dir.filePath(stations[i]).toAscii();
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Load this station.
        error = oskar_telescope_model_load_private(telescope, log, settings,
                station_name, s, depth + 1, data_files, element_models);
        if (error) return error;
    }

    return OSKAR_SUCCESS;
}


static void oskar_telescope_model_get_data_files(
        QHash<QString, QString>& data_files, const QDir& dir, int depth)
{
    // Check for CST element pattern data.
    if (dir.exists(element_x_name_cst))
    {
        data_files[QString(element_x_name_cst)] = dir.absoluteFilePath(element_x_name_cst);
    }
    if (dir.exists(element_y_name_cst))
    {
        data_files[QString(element_y_name_cst)] = dir.absoluteFilePath(element_y_name_cst);
    }
 }


static int oskar_telescope_model_load_layout(
        oskar_TelescopeModel* telescope,
        const oskar_SettingsTelescope* settings, const QDir& dir,
        int num_station_dirs)
{
    int error = OSKAR_SUCCESS;

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
        // TODO
        return OSKAR_ERR_SETUP_FAIL;
    }

    return error;
}


static int oskar_telescope_model_load_station_config(oskar_StationModel* station,
        const QDir& dir)
{
    int error = OSKAR_SUCCESS;

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

    return error;
}


static int oskar_telescope_model_allocate_children(oskar_StationModel* station,
        int num_station_dirs, int type)
{
    int error = OSKAR_SUCCESS;

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
                type, OSKAR_LOCATION_CPU, 0);
        if (error) return error;
    }

    return error;
}


static int oskar_telescope_model_load_element_patterns(oskar_Log* log,
        const oskar_SettingsTelescope* settings,
        oskar_StationModel* station, QHash<QString, QString> data_files,
        QHash<QString, oskar_ElementModel*>& models)
{
    int error = OSKAR_SUCCESS;

    QByteArray element_x, element_y;
    const char *element_file_x = NULL, *element_file_y = NULL;

    if (data_files.contains(QString(element_x_name_cst)))
    {
        element_x = data_files.value(QString(element_x_name_cst)).toAscii();
        element_file_x = element_x.constData();
    }
    if (data_files.contains(QString(element_y_name_cst)))
    {
        element_y = data_files.value(QString(element_y_name_cst)).toAscii();
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
                error = oskar_element_model_load_cst(station->element_pattern,
                        log, 1, element_file_x, &settings->station.element_fit);
                if (error) return error;
            }
            if (element_file_y)
            {

                oskar_log_message(log, 0, "Loading CST element "
                        "pattern data (Y): %s", element_file_y);
                oskar_log_message(log, 0, "");
                error = oskar_element_model_load_cst(station->element_pattern,
                        log, 2, element_file_y, &settings->station.element_fit);
                if (error) return error;
            }

            // Store pointer to the element model for these files.
            models.insert(files, station->element_pattern);
        }
    }

    return error;
}
