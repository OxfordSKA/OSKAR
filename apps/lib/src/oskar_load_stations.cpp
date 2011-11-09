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


#include "apps/lib/oskar_load_stations.h"
#include "utility/oskar_load_csv_coordinates_2d.h" // Deprecated.
#include "utility/oskar_mem_element_size.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QFileInfoList>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_load_stations(oskar_StationModel* station, int* identical_stations,
        const int num_stations, const char* dir_path)
{
    // Get the list of station files to load.
    QDir dir;
    dir.setPath(QString(dir_path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");

    // Check that the number of stations is the same as that supplied.
    if (num_stations != files.size())
        return OSKAR_ERR_DIMENSION_MISMATCH;

    // Check that the data is in the right location.
    for (int i = 0; i < num_stations; ++i)
    {
        if (station[i].location() != OSKAR_LOCATION_CPU)
            return OSKAR_ERR_BAD_LOCATION;
    }

    // Load the station data.
    for (int i = 0; i < num_stations; ++i)
    {
        const char* filename = files.at(i).absoluteFilePath().toLatin1().data();
        station[i].load(filename);
    }

    // Check if stations are all the same.
    *identical_stations = true;

    // 1. Check if they have the same number of antennas.
    int num_antennas_station0 = station[0].num_elements;
    for (int i = 0; i < num_stations; ++i)
    {
        if (station[i].num_elements != num_antennas_station0)
        {
            *identical_stations = false;
            break;
        }
    }

    // 2. Check if the positions are the same (compare every byte).
    bool done = false;
    oskar_StationModel* station0 = &station[0];
    if (*identical_stations)
    {
        for (int j = 0; j < num_stations; ++j)
        {
            oskar_StationModel* s = &station[j];
            int bytes = s->num_elements * oskar_mem_element_size(s->type());
            for (int i = 0; i < bytes; ++i)
            {
                if (((char*)(station0->x))[i] != ((char*)(s->x))[i] ||
                        ((char*)(station0->y))[i] != ((char*)(s->y))[i] ||
                        ((char*)(station0->z))[i] != ((char*)(s->z))[i])
                {
                    *identical_stations = false;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }
    return 0;
}


// DEPRECATED
unsigned oskar_load_stations_d(const char* dir_path, oskar_StationModel_d** stations,
        int* identical_stations)
{
    int num_stations = 0;
    QDir dir;
    dir.setPath(QString(dir_path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");
    num_stations = files.size();
    *stations = (oskar_StationModel_d*) malloc(num_stations * sizeof(oskar_StationModel_d));

    for (int i = 0; i < num_stations; ++i)
    {
        oskar_StationModel_d * s = &(*stations)[i];
        const char * filename = files.at(i).absoluteFilePath().toLatin1().data();
        oskar_load_csv_coordinates_2d_d(filename, &s->num_antennas,
                &s->antenna_x, &s->antenna_y);
    }

    // Check if stations are all the same.
    *identical_stations = true;
    // 1. Check if they have the same number of antennas.
    int num_antennas_station0 = (*stations)[0].num_antennas;
    for (int j = 0; j < num_stations; ++j)
    {
        oskar_StationModel_d * s = &(*stations)[j];
        if ((int)s->num_antennas != num_antennas_station0)
        {
            *identical_stations = false;
            break;
        }
    }

    // 2. Check if the positions are are the same.
    bool done = false;
    oskar_StationModel_d* station0 = &(*stations)[0];
    if (*identical_stations)
    {
        for (int j = 0; j < num_stations; ++j)
        {
            oskar_StationModel_d * s = &(*stations)[j];
            for (int i = 0; i < (int)s->num_antennas; ++i)
            {
                if (station0->antenna_x[i] != s->antenna_x[i] ||
                        station0->antenna_y[i] != s->antenna_y[i])
                {
                    *identical_stations = false;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }

    return num_stations;
}




// DEPRECATED
unsigned oskar_load_stations_f(const char* dir_path, oskar_StationModel_f** stations,
        int* identical_stations)
{
    int num_stations = 0;
    QDir dir;
    dir.setPath(QString(dir_path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");
    num_stations = files.size();
    *stations = (oskar_StationModel_f*) malloc(num_stations * sizeof(oskar_StationModel_f));

    for (int i = 0; i < num_stations; ++i)
    {
        oskar_StationModel_f * s = &(*stations)[i];
        const char * filename = files.at(i).absoluteFilePath().toLatin1().data();
        oskar_load_csv_coordinates_2d_f(filename, &s->num_antennas,
                &s->antenna_x, &s->antenna_y);
    }

    // Check if stations are all the same.
    *identical_stations = true;
    // 1. Check if they have the same number of antennas.
    int num_antennas_station0 = (*stations)[0].num_antennas;
    for (int j = 0; j < num_stations; ++j)
    {
        oskar_StationModel_f * s = &(*stations)[j];
        if ((int)s->num_antennas != num_antennas_station0)
        {
            *identical_stations = false;
            break;
        }
    }

    // 2. Check if the positions are are the same.
    bool done = false;
    oskar_StationModel_f* station0 = &(*stations)[0];
    if (*identical_stations)
    {
        for (int j = 0; j < num_stations; ++j)
        {
            oskar_StationModel_f * s = &(*stations)[j];
            for (int i = 0; i < (int)s->num_antennas; ++i)
            {
                if (station0->antenna_x[i] != s->antenna_x[i] ||
                        station0->antenna_y[i] != s->antenna_y[i])
                {
                    *identical_stations = false;
                    done = true;
                    break;
                }
            }
            if (done) break;
        }
    }

    return num_stations;
}

#ifdef __cplusplus
}
#endif
