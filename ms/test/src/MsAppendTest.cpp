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

#include "ms/test/MsAppendTest.h"
#include "ms/MsCreate.h"
#include "ms/MsAppend.h"
#include "ms/oskar_ms_create_empty.h"
#include "ms/oskar_ms_create_meta1.h"
#include "ms/oskar_ms_append_vis1.h"

using namespace oskar;

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(MsAppendTest);

/**
 * @details
 * Sets up the context before running each test method.
 */
void MsAppendTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void MsAppendTest::tearDown()
{
}

/**
 * @details
 * Tests appending visibilities to a measurement set.
 */
void MsAppendTest::test()
{
}

/**
 * @details
 * Tests appending to a measurement set using the C binding.
 */
void MsAppendTest::test_c()
{
    // Define filename and metadata.
    const char filename[] = "append_c.ms";
    double mjd = 2455632.20209 - 2400000.5;
    double exposure = 90;
    double interval = 90;
    double ra = 0;
    double dec = 1.570796;
    double freq = 400e6;

    // Define antenna positions.
    float ax[] = {0, 0, 0};
    float ay[] = {0, 0, 0};
    float az[] = {0, 0, 0};
    int na = sizeof(ax) / sizeof(float);

    // Define visibilities.
    float u[] = {1000.0, 2000.01, 156.03};
    float v[] = {0.0, -241.02, 1678.04};
    float w[] = {0.0, -56.0, 145.0};
    float vis[] = {1.0, 0.0, 0.00, 0.0, 0.00, 0.0};
    int ant1[] = {0, 0, 1};
    int ant2[] = {1, 2, 2};
    int nv = sizeof(u) / sizeof(float);

    oskar_ms_create_meta1(filename, mjd, ra, dec, na, ax, ay, az, freq);
    oskar_ms_append_vis1(filename, mjd, exposure, interval,
            nv, u, v, w, vis, ant1, ant2);
}
