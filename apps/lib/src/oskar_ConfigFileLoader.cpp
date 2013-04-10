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

#include "apps/lib/oskar_ConfigFileLoader.h"
#include "interferometry/oskar_telescope_model_load_station_coords.h"
#include "station/oskar_station_model_init_child_stations.h"
#include "station/oskar_station_model_load_config.h"
#include "station/oskar_station_model_resize.h"
#include "station/oskar_station_model_resize_element_types.h"
#include "station/oskar_station_model_set_element_coords.h"
#include "station/oskar_station_model_set_element_errors.h"
#include "station/oskar_station_model_set_element_orientation.h"
#include "station/oskar_station_model_set_element_weight.h"

#include <QtCore/QDir>

const char oskar_ConfigFileLoader::config_file[] = "config.txt";
const char oskar_ConfigFileLoader::layout_file[] = "layout.txt";

oskar_ConfigFileLoader::oskar_ConfigFileLoader(const oskar_Settings* settings)
{
    settings_ = settings;
}

oskar_ConfigFileLoader::~oskar_ConfigFileLoader()
{
}

void oskar_ConfigFileLoader::load(oskar_TelescopeModel* telescope,
        const QDir& cwd, int num_subdirs, QHash<QString, QString>& /*filemap*/,
        int* status)
{
    // Check for presence of "config.txt".
    const char* file = NULL;
    if (cwd.exists(layout_file))
        file = layout_file;
    else if (cwd.exists(config_file))
        file = config_file;
    else
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;

    // Load the interferometer layout.
    oskar_telescope_model_load_station_coords(telescope,
            cwd.filePath(file).toAscii(), settings_->telescope.longitude_rad,
            settings_->telescope.latitude_rad,
            settings_->telescope.altitude_m, status);

    // If no subdirectories, set all "stations" to be a single dipole.
    if (num_subdirs == 0)
    {
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
}

void oskar_ConfigFileLoader::load(oskar_StationModel* station, const QDir& cwd,
        int num_subdirs, int /*depth*/, QHash<QString, QString>& /*filemap*/,
        int* status)
{
    // Check for presence of "config.txt".
    if (cwd.exists(config_file))
    {
        oskar_station_model_load_config(station,
                cwd.filePath(config_file).toAscii(), status);
    }
    else
        *status = OSKAR_ERR_SETUP_FAIL_TELESCOPE_CONFIG_FILE_MISSING;

    // Check if this is the last level.
    if (num_subdirs > 0)
    {
        // Allocate storage for child stations.
        oskar_station_model_init_child_stations(station, status);
    }
    else if (num_subdirs == 0)
    {
        // Allocate storage for element model data if no children.
        oskar_station_model_resize_element_types(station, 1, status);
    }
}

