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

#include "apps/lib/oskar_remove_dir.h"
#include "apps/lib/oskar_telescope_model_save.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_telescope_model_save_station_coords.h"
#include "station/oskar_station_model_save_config.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>

static const char config_name[] = "config.txt";

static int oskar_telescope_model_save_private(const oskar_TelescopeModel* telescope,
        const char* dir_path, oskar_StationModel* station, int depth);

extern "C"
int oskar_telescope_model_save(const oskar_TelescopeModel* telescope,
        const char* dir_path)
{
    return oskar_telescope_model_save_private(telescope, dir_path, NULL, 0);
}

static int oskar_telescope_model_save_private(const oskar_TelescopeModel* telescope,
        const char* dir_path, oskar_StationModel* station, int depth)
{
    int error = 0, num_stations = 0;

    if (depth == 0)
    {
        // Check if directory already exists, and remove it if so.
        QDir dir;
        dir.setPath(dir_path);
        if (dir.exists())
        {
            if (!oskar_remove_dir(dir_path))
                return OSKAR_ERR_FILE_IO;
        }
    }

    // Create the directory if it doesn't exist.
    QDir dir;
    dir.setPath(dir_path);
    if (!dir.exists())
    {
        QDir temp;
        temp.mkpath(QString(dir_path));
    }

    if (depth == 0)
    {
        // Write the station coordinates.
        QByteArray coord_path = dir.filePath(config_name).toAscii();
        oskar_telescope_model_save_station_coords(telescope, coord_path,
                &error);
        if (error) return error;

        // Get the number of stations.
        num_stations = telescope->num_stations;
    }
    else
    {
        // Write the station configuration.
        QByteArray config_path = dir.filePath(config_name).toAscii();
        error = oskar_station_model_save_config(config_path, station);
        if (error) return error;

        // Get the number of stations.
        if (station->child)
            num_stations = station->num_elements;
    }

    // Recursive call to write stations.
    for (int i = 0; i < num_stations; ++i)
    {
        // Get the name of the station, and a pointer to the station to save.
        QByteArray station_name = dir.filePath(QString("level%1_%2").
                arg(depth).arg(i, 3, 10, QChar('0'))).toAscii();
        oskar_StationModel* s;
        s = (depth == 0) ? &telescope->station[i] : &station->child[i];

        // Save this station.
        error = oskar_telescope_model_save_private(telescope, station_name,
                s, depth + 1);
    }

    return error;
}
