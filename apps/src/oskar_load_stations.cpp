/*
 * Copyright (c) 2011, The University of Oxford
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


#include "apps/oskar_load_stations.h"

#include "utility/oskar_load_csv_coordinates_2d.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QFileInfoList>

#include <cstdio>
#include <cstdlib>
#include <vector>

unsigned oskar_load_stations(const char* dir_path, oskar_StationModel** stations,
        bool* idential_stations)
{
    int num_stations = 0;
    QDir dir;
    dir.setPath(QString(dir_path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");
    num_stations = files.size();
    *stations = (oskar_StationModel*) malloc(num_stations * sizeof(oskar_StationModel));

    for (int i = 0; i < num_stations; ++i)
    {
        oskar_StationModel * s = &(*stations)[i];
        const char * filename = files.at(i).absoluteFilePath().toLatin1().data();
        oskar_load_csv_coordinates_2d(filename, &s->num_antennas,
                &s->antenna_x, &s->antenna_y);
    }

    // Check if stations are all the same.
    *idential_stations = true;
    // 1. Check if they have the same number of antennas.
    int num_antennas_station0 = (*stations)[0].num_antennas;
    for (int j = 0; j < num_stations; ++j)
    {
        oskar_StationModel * s = &(*stations)[j];
        if ((int)s->num_antennas != num_antennas_station0)
        {
            *idential_stations = false;
            break;
        }
    }

    // 2. Check if the positions are are the same.
    bool done = false;
    oskar_StationModel* station0 = &(*stations)[0];
    if (*idential_stations)
    {
        for (int j = 0; j < num_stations; ++j)
        {
            oskar_StationModel * s = &(*stations)[j];
            for (int i = 0; i < (int)s->num_antennas; ++i)
            {
                if (station0->antenna_x[i] != s->antenna_x[i] ||
                        station0->antenna_y[i] != s->antenna_y[i])
                {
                    *idential_stations = false;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }

    return num_stations;
}

