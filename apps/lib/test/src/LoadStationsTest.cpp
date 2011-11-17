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

#include "apps/lib/test/LoadStationsTest.h"
#include "apps/lib/oskar_load_stations.h"
#include "station/oskar_StationModel.h"
#include "station/oskar_station_model_free.h"
#include "station/oskar_station_model_init.h"

#include <QtCore/QFile>
#include <QtCore/QDir>
#include <QtCore/QStringList>
#include <QtCore/QTextStream>
#include <QtCore/QFileInfo>
#include <QtCore/QFileInfoList>

#include <cstdio>

void LoadStationsTest::test_load()
{
    const char* path = "temp_test_stations";

    // Create temp station directory.
    int num_stations = 25;
    int num_antennas = 10;
    QDir dir;
    dir.mkdir(QString(path));
    for (int j = 0; j < num_stations; ++j)
    {
        QString station_name = "station_" +
                QString("0000" + QString::number(j)).right(4) + ".dat";
        QFile file(QString(path) + QDir::separator() + station_name);
        file.open(QIODevice::WriteOnly);
        QTextStream out(&file);
        for (int i = 0; i < num_antennas; ++i)
        {
            out << (float)i + i/10.0 << "," << (float)i - i/10.0 << endl;
        }
        file.flush();
        file.close();
    }

    // Load the stations.
    oskar_StationModel* stations = (oskar_StationModel*) malloc(num_stations * sizeof(oskar_StationModel));
    for (int i = 0; i < num_stations; ++i)
        oskar_station_model_init(&stations[i], OSKAR_SINGLE, OSKAR_LOCATION_CPU, 0);
    int identical_stations;
    int error = oskar_load_stations(stations, &identical_stations, num_stations, path);
    CPPUNIT_ASSERT_EQUAL(0, error);

    // Check the data loaded correctly.
    double err = 1.0e-6;
    for (int j = 0; j < num_stations; ++j)
    {
        CPPUNIT_ASSERT_EQUAL(num_antennas, (int)stations[j].num_elements);
        for (int i = 0; i < num_antennas; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i + i/10.0, ((float*)(stations[j].x))[i], err);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i - i/10.0, ((float*)(stations[j].y))[i], err);
        }
    }
    CPPUNIT_ASSERT_EQUAL(1, identical_stations);

    // Remove the test directory.
    dir.setPath(QString(path));
    QFileInfoList files = dir.entryInfoList(QStringList() << "*.dat");
    for (int i = 0; i < files.size(); ++i)
        QFile::remove(files.at(i).absoluteFilePath());
    dir.rmdir(dir.absolutePath());

    // Free the stations.
    for (int i = 0; i < num_stations; ++i)
        oskar_station_model_free(&stations[i]);
    free(stations);
}
