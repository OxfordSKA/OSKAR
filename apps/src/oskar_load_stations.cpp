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

#include "utility/oskar_load_csv_coordinates.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QFile>
#include <QtCore/QFileInfo>
#include <QtCore/QFileInfoList>

#include <cstdio>
#include <cstdlib>

unsigned oskar_load_stations(const char* path, oskar_StationModel** stations)
{
    int num_stations = 0;
    QDir dir;
    dir.setPath(QString(path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");
    num_stations = files.size();
    *stations = (oskar_StationModel*) malloc(num_stations * sizeof(oskar_StationModel));

    for (int i = 0; i < files.size(); ++i)
    {
        const char * filename = files.at(i).absoluteFilePath().toLatin1().data();
        double*  ax = (*stations)[i].antenna_x;
        double*  ay = (*stations)[i].antenna_y;
        unsigned* n = &((*stations)[i].num_antennas);
        oskar_load_csv_coordinates(filename, n, &ax, &ay);
    }

    return num_stations;
}

