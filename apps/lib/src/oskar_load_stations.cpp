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
#include "utility/oskar_mem_element_size.h"

#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QFileInfoList>

extern "C"
int oskar_load_stations(oskar_StationModel* station, int num_stations,
        const char* dir_path)
{
    // Get the list of station files to load.
    QDir dir;
    dir.setPath(QString(dir_path));
    if (!dir.exists())
        return OSKAR_ERR_FILE_IO;
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
        QByteArray filename = files.at(i).absoluteFilePath().toAscii();
        station[i].load(filename);
    }

    return 0;
}
