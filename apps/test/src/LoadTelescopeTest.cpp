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

#include "apps/test/LoadTelescopeTest.h"
#include "apps/oskar_load_telescope.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_horizon_plane_to_itrs.h"

#include <QtCore/QFile>
#include <QtCore/QStringList>
#include <QtCore/QTextStream>

#include <cstdio>

void LoadTelescopeTest::test_load()
{
    // Generate some test data.
    const char* filename = "temp_telescope.dat";
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text))
        return;
    QTextStream out(&file);
    int num_stations = 10;
    double* x_horizon = new double[num_stations];
    double* y_horizon = new double[num_stations];
    for (int i = 0; i < num_stations; ++i)
    {
        x_horizon[i] = (float)i + i/10.0;
        y_horizon[i] = (float)i - i/10.0;
        out << x_horizon[i] << "," << y_horizon[i] << endl;
    }
    file.close();


    // Load the telescope file.
    double longitude_rad = 0.0;
    double latitude_rad  = 0.0;
    oskar_TelescopeModel telescope;
    oskar_load_telescope(filename, longitude_rad, latitude_rad, &telescope);

    // Convert input test horizon coordinates to ITRS.
    double* x_itrs = new double[num_stations];
    double* y_itrs = new double[num_stations];
    double* z_itrs = new double[num_stations];
    oskar_horizon_plane_to_itrs(num_stations, x_horizon, y_horizon, latitude_rad,
            x_itrs, y_itrs, z_itrs);

    // Check the data loaded correctly.
    CPPUNIT_ASSERT_EQUAL(num_stations, (int)telescope.num_antennas);
    for (int i = 0; i < num_stations; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(x_itrs[i], telescope.antenna_x[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(y_itrs[i], telescope.antenna_y[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(z_itrs[i], telescope.antenna_z[i], 1.0e-6);
    }

    // Cleanup.
    delete[] x_horizon;
    delete[] y_horizon;
    delete[] x_itrs;
    delete[] y_itrs;
    delete[] z_itrs;

    QFile::remove(filename);
}
