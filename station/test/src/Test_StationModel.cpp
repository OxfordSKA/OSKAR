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

#include "station/test/Test_StationModel.h"
#include "station/oskar_station_model_load.h"
#include "station/oskar_StationModel.h"
#include "utility/oskar_vector_types.h"

#include <cmath>
#include <cstdio>

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Sets up the context before running each test method.
 */
void Test_StationModel::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void Test_StationModel::tearDown()
{
}

/**
 * @details
 * Tests loading of station data.
 */
void Test_StationModel::test_load_single()
{
    // Create the test file.
    const char* filename = "test.dat";
    FILE* file = fopen(filename, "w");
    int n_elements = 100;
    for (int i = 0; i < n_elements/2; ++i)
        fprintf(file, "%.3f %.3f %.3f\n", i/10.0, i/20.0, i/30.0);
    fprintf(file, "\n"); // Add a blank line halfway.
    for (int i = n_elements/2; i < n_elements; ++i)
        fprintf(file, "%.3f,%.3f,%.3f\n", i/10.0, i/20.0, i/30.0);
    fclose(file);

    // Load the data.
    oskar_StationModel station_model(OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    oskar_station_model_load(&station_model, filename);

    // Check the coordinates.
    CPPUNIT_ASSERT_EQUAL(n_elements, station_model.num_elements);
    for (int i = 0; i < n_elements; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0,
                ((float*)station_model.x_weights.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0,
                ((float*)station_model.y_weights.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/30.0,
                ((float*)station_model.z_weights.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0,
                ((float2*)station_model.weight.data)[i].x, 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((float2*)station_model.weight.data)[i].y, 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0,
                ((float*)station_model.amp_gain.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((float*)station_model.amp_gain_error.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((float*)station_model.phase_offset.data)[i], 1e-3);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((float*)station_model.phase_error.data)[i], 1e-3);
    }

    // Remove the test file.
    remove(filename);
}

/**
 * @details
 * Tests loading of station data.
 */
void Test_StationModel::test_load_double()
{
    // Create the test file.
    const char* filename = "test.dat";
    FILE* file = fopen(filename, "w");
    int n_elements = 100;
    for (int i = 0; i < n_elements/2; ++i)
        fprintf(file, "%.6f %.6f %.6f\n", i/10.0, i/20.0, i/30.0);
    fprintf(file, "\n"); // Add a blank line halfway.
    for (int i = n_elements/2; i < n_elements; ++i)
        fprintf(file, "%.6f,%.6f,%.6f\n", i/10.0, i/20.0, i/30.0);
    fclose(file);

    // Load the data.
    oskar_StationModel station_model(OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
    oskar_station_model_load(&station_model, filename);

    // Check the coordinates.
    CPPUNIT_ASSERT_EQUAL(n_elements, station_model.num_elements);
    for (int i = 0; i < n_elements; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0,
                ((double*)station_model.x_weights.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0,
                ((double*)station_model.y_weights.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/30.0,
                ((double*)station_model.z_weights.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0,
                ((double2*)station_model.weight.data)[i].x, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((double2*)station_model.weight.data)[i].y, 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0,
                ((double*)station_model.amp_gain.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((double*)station_model.amp_gain_error.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((double*)station_model.phase_offset.data)[i], 1e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0,
                ((double*)station_model.phase_error.data)[i], 1e-6);
    }

    // Remove the test file.
    remove(filename);
}
