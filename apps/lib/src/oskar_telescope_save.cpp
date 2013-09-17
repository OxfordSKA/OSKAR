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

#include "apps/lib/oskar_remove_dir.h"
#include "apps/lib/oskar_telescope_save.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>

static const char config_name[] = "config.txt";

static void oskar_telescope_save_private(const oskar_Telescope* telescope,
        const char* dir_path, const oskar_Station* station, int depth,
        int* status);

extern "C"
void oskar_telescope_save(const oskar_Telescope* telescope,
        const char* dir_path, int* status)
{
    oskar_telescope_save_private(telescope, dir_path, NULL, 0, status);
}

static void oskar_telescope_save_private(const oskar_Telescope* telescope,
        const char* dir_path, const oskar_Station* station, int depth,
        int* status)
{
    int num_stations = 0;

    if (depth == 0)
    {
        // Check if directory already exists, and remove it if so.
        QDir dir;
        dir.setPath(dir_path);
        if (dir.exists())
        {
            if (!oskar_remove_dir(dir_path))
            {
                *status = OSKAR_ERR_FILE_IO;
                return;
            }
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
        QByteArray coord_path = dir.filePath(config_name).toLatin1();
        oskar_telescope_save_station_coords(telescope, coord_path,
                status);

        // Get the number of stations.
        num_stations = oskar_telescope_num_stations(telescope);
    }
    else
    {
        // Write the station configuration.
        QByteArray config_path = dir.filePath(config_name).toLatin1();
        oskar_station_save_config(config_path, station, status);

        // Get the number of stations.
        if (oskar_station_has_child(station))
            num_stations = oskar_station_num_elements(station);
    }

    // Recursive call to write stations.
    for (int i = 0; i < num_stations; ++i)
    {
        // Get the name of the station, and a pointer to the station to save.
        QByteArray station_name = dir.filePath(QString("level%1_%2").
                arg(depth).arg(i, 3, 10, QChar('0'))).toLatin1();
        const oskar_Station* s;
        s = (depth == 0) ? oskar_telescope_station_const(telescope, i) :
                oskar_station_child_const(station, i);

        // Save this station.
        oskar_telescope_save_private(telescope, station_name,
                s, depth + 1, status);
    }
}
