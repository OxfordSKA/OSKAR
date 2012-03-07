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

#include "interferometry/test/Test_TelescopeModel.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "interferometry/oskar_horizon_plane_to_offset_geocentric_cartesian.h"
#include <cmath>
#include <cstdio>

static const char* telescope_file_name = "test_telescope.dat";
static const char* station_base = "test_station";
static const int n_stations = 25;
static const int n_elements = 200;

/**
 * @details
 * Sets up the context before running each test method.
 */
void Test_TelescopeModel::setUp()
{
    // Create a telescope coordinate file.
    FILE* file;
    char station_name[80];
    file = fopen(telescope_file_name, "w");
    for (int i = 0; i < n_stations; ++i)
        fprintf(file, "%.8f,%.8f,%.8f\n", i / 10.0, i / 20.0, i / 30.0);
    fclose(file);

    // Create some station coordinate files.
    for (int i = 0; i < n_stations; ++i)
    {
        sprintf(station_name, "%s_%d.dat", station_base, i);
        file = fopen(station_name, "w");
        for (int j = 0; j < n_elements; ++j)
        {
            int t = j + i;
            fprintf(file, "%.8f,%.8f,%.8f\n", t / 5.0, t / 6.0, t / 7.0);
        }
        fclose(file);
    }
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void Test_TelescopeModel::tearDown()
{
    char station_name[80];

    // Remove telescope coordinate file.
    remove(telescope_file_name);

    // Remove station coordinate files.
    for (int i = 0; i < n_stations; ++i)
    {
        sprintf(station_name, "%s_%d.dat", station_base, i);
        remove(station_name);
    }
}

/**
 * @details
 * Tests filling a telescope model, and copying it to the GPU and back.
 */
void Test_TelescopeModel::test_load_telescope_cpu()
{
    char station_name[80];

    // Set the location.
    double longitude = 30.0 * M_PI / 180.0;
    double latitude = 50.0 * M_PI / 180.0;
    double altitude = 0.0;

    try
    {
        oskar_TelescopeModel* tel_cpu = new oskar_TelescopeModel(OSKAR_DOUBLE,
                OSKAR_LOCATION_CPU);

        // Fill the telescope structure.
        CPPUNIT_ASSERT_EQUAL(0, tel_cpu->load_station_coords(telescope_file_name,
                longitude, latitude, altitude));
        for (int i = 0; i < n_stations; ++i)
        {
            sprintf(station_name, "%s_%d.dat", station_base, i);
            CPPUNIT_ASSERT_EQUAL(0, tel_cpu->load_station(i, station_name));
            CPPUNIT_ASSERT_EQUAL(n_elements, tel_cpu->station[i].num_elements);
        }

        // Copy telescope structure to GPU.
        oskar_TelescopeModel* tel_gpu = new oskar_TelescopeModel(tel_cpu,
                OSKAR_LOCATION_GPU);

        // Delete the old CPU structure.
        delete tel_cpu;
        tel_cpu = NULL;

        // Copy the telescope structure back to the CPU.
        tel_cpu = new oskar_TelescopeModel(tel_gpu, OSKAR_LOCATION_CPU);

        // Delete the old GPU structure.
        delete tel_gpu;

        // Check the contents of the CPU structure.
        for (int i = 0; i < n_stations; ++i)
        {
            // Define horizon coordinates.
            double x_hor = i / 10.0;
            double y_hor = i / 20.0;
            double z_hor = i / 30.0;

            // Compute offset geocentric coordinates.
            double x = 0.0, y = 0.0, z = 0.0;
            oskar_horizon_plane_to_offset_geocentric_cartesian_d(1,
                    &x_hor, &y_hor, &z_hor, longitude, latitude, &x, &y, &z);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(x, ((double*)(tel_cpu->station_x))[i], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(y, ((double*)(tel_cpu->station_y))[i], 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(z, ((double*)(tel_cpu->station_z))[i], 1e-5);

            for (int j = 0; j < n_elements; ++j)
            {
                int t = j + i;
                double x = t / 5.0;
                double y = t / 6.0;
                double z = t / 7.0;
                CPPUNIT_ASSERT_DOUBLES_EQUAL(x, ((double*)(tel_cpu->station[i].x_weights))[j], 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(y, ((double*)(tel_cpu->station[i].y_weights))[j], 1e-5);
                CPPUNIT_ASSERT_DOUBLES_EQUAL(z, ((double*)(tel_cpu->station[i].z_weights))[j], 1e-5);
            }
        }

        // Delete the CPU structure.
        delete tel_cpu;
    }
    catch (const char* msg)
    {
        CPPUNIT_FAIL(msg);
    }
}
