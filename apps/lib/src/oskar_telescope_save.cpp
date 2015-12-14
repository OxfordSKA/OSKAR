/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "apps/lib/oskar_dir.h"
#include "apps/lib/oskar_telescope_save.h"
#include <apps/lib/oskar_dir.h>

#include <QtCore/QDir>
#include <QtCore/QStringList>

static const char layout_name[] = "layout.txt";
static const char apodisaion_name[] = "apodisation.txt";
static const char feed_x_name[] = "feed_angle_x.txt";
static const char feed_y_name[] = "feed_angle_y.txt";
static const char element_types_name[] = "element_types.txt";
static const char mount_types_name[] = "mount_types.txt";
static const char gain_phase_name[] = "gain_phase.txt";
static const char permitted_beams_name[] = "permitted_beams.txt";

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
        if (oskar_dir_exists(dir_path))
        {
            if (!oskar_dir_remove(dir_path))
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
        QByteArray path = dir.filePath(layout_name).toLatin1();
        oskar_telescope_save_layout(telescope, path, status);

        // Get the number of stations.
        num_stations = oskar_telescope_num_stations(telescope);
    }
    else
    {
        // Write the station configuration data.
        QByteArray path;
        path = dir.filePath(layout_name).toLatin1();
        oskar_station_save_layout(path, station, status);
        path = dir.filePath(feed_x_name).toLatin1();
        oskar_station_save_feed_angle(path, station, 1, status);
        path = dir.filePath(feed_y_name).toLatin1();
        oskar_station_save_feed_angle(path, station, 0, status);
        path = dir.filePath(mount_types_name).toLatin1();
        oskar_station_save_mount_types(path, station, status);
        if (oskar_station_apply_element_errors(station))
        {
            path = dir.filePath(gain_phase_name).toLatin1();
            oskar_station_save_gain_phase(path, station, status);
        }
        if (oskar_station_apply_element_weight(station))
        {
            path = dir.filePath(apodisaion_name).toLatin1();
            oskar_station_save_apodisation(path, station, status);
        }
        if (oskar_station_num_element_types(station) > 1)
        {
            path = dir.filePath(element_types_name).toLatin1();
            oskar_station_save_element_types(path, station, status);
        }
        if (oskar_station_num_permitted_beams(station) > 0)
        {
            path = dir.filePath(permitted_beams_name).toLatin1();
            oskar_station_save_permitted_beams(path, station, status);
        }

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
