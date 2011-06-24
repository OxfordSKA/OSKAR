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

#include "ms/test/MsCreateTest.h"
#include "ms/MsCreate.h"
#include "ms/oskar_ms_create_vis1.h"

using namespace oskar;

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(MsCreateTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void MsCreateTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void MsCreateTest::tearDown()
{
}

/**
 * @details
 * Tests creation of simple measurement set.
 */
void MsCreateTest::test()
{
    // Create the MsCreate object, passing it the filename.
    MsCreate ms("simple.ms", (2455632.20209 - 2400000.5), 90, 90);

    // Add some dummy antenna positions.
    double ax[] = {0, 0, 0};
    double ay[] = {0, 0, 0};
    double az[] = {0, 0, 0};
    int na = sizeof(ax) / sizeof(double);
    ms.addAntennas(na, ax, ay, az);

    // Add the Right Ascension & Declination of field centre.
    ms.addField(0, 1.570796);

    // Add frequency band.
    ms.addBand(0, 1, 400e6, 1.0);

    // Add test visibilities (don't include conjugated versions).
    double u[] = {1000.0, 2000.01, 156.03};
    double v[] = {0.0, -241.02, 1678.04};
    double w[] = {0.0, -56.0, 145.0};
    double vis[] = {1.0, 0.0, 0.00, 0.0, 0.00, 0.0};
    int ant1[] = {0, 0, 1};
    int ant2[] = {1, 2, 2};
    int nv = sizeof(u) / sizeof(double);
    ms.addVisibilities(1, nv, u, v, w, vis, ant1, ant2);
}

/**
 * @details
 * Tests creation of simple measurement set using the C binding.
 */
void MsCreateTest::test_c()
{
    // Define filename and metadata.
    const char filename[] = "simple_c.ms";
    double mjd = 2455632.20209 - 2400000.5;
    double exposure = 90;
    double interval = 90;
    double ra = 0;
    double dec = 1.570796;
    double freq = 400e6;

    // Define antenna positions.
    double ax[] = {0, 0, 0};
    double ay[] = {0, 0, 0};
    double az[] = {0, 0, 0};
    int na = sizeof(ax) / sizeof(double);

    // Define visibilities.
    double u[] = {1000.0, 2000.01, 156.03};
    double v[] = {0.0, -241.02, 1678.04};
    double w[] = {0.0, -56.0, 145.0};
    double vis[] = {1.0, 0.0, 0.00, 0.0, 0.00, 0.0};
    int ant1[] = {0, 0, 1};
    int ant2[] = {1, 2, 2};
    int nv = sizeof(u) / sizeof(double);

    oskar_ms_create_vis1(filename, mjd, exposure, interval, ra, dec,
            na, ax, ay, az, nv, u, v, w, vis, ant1, ant2, freq);
}
