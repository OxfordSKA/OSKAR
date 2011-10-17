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

#include "utility/test/LoadCoordinatesTest.h"
#include "utility/oskar_load_csv_coordinates_2d.h"
#include "utility/oskar_load_csv_coordinates_3d.h"

//#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cstdio>
#include <cstdlib>

void LoadCoordinatesTest::test_load_2d()
{
    const char* filename = "temp_coordinates.dat";
    FILE* file = fopen(filename, "w");
    if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
    int num_coords = 100000;
    for (int i = 0; i < num_coords; ++i)
    {
        fprintf(file, "%f,%f\n", i/10.0, i/20.0);
    }
    fclose(file);

    double* x = NULL;
    double* y = NULL;
    unsigned n;
    TIMER_START
    oskar_load_csv_coordinates_2d_d(filename, &n, &x, &y);
    TIMER_STOP("Loaded %d 2D coordinate pairs", n)

    // Check the data loaded correctly.
    CPPUNIT_ASSERT_EQUAL(num_coords, (int)n);
    for (int i = 0; i < num_coords; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0, x[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0, y[i], 1.0e-6);
    }

    // Cleanup.
    remove(filename);
}

void LoadCoordinatesTest::test_load_3d()
{
    const char* filename = "temp_coordinates.dat";
    FILE* file = fopen(filename, "w");
    if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
    int num_coords = 100000;
    for (int i = 0; i < num_coords; ++i)
    {
        fprintf(file, "%f,%f,%f\n", i/10.0, i/20.0, i/30.0);
    }
    fclose(file);

    double* x = NULL;
    double* y = NULL;
    double* z = NULL;
    unsigned n;
    TIMER_START
    oskar_load_csv_coordinates_3d_d(filename, &n, &x, &y, &z);
    TIMER_STOP("Loaded %d 3D coordinate pairs", n)

    // Check the data loaded correctly.
    CPPUNIT_ASSERT_EQUAL(num_coords, (int)n);
    for (int i = 0; i < num_coords; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0, x[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0, y[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/30.0, z[i], 1.0e-6);
    }

    // Cleanup.
    remove(filename);
}
