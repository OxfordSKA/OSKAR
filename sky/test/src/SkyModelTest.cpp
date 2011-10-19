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

#include "sky/test/SkyModelTest.h"
#include "sky/oskar_SkyModel.h"
#include "sky/oskar_sky_model_load.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cstdio>
#include <cstdlib>

void SkyModelTest::test_load()
{
    const char* filename = "temp_sources.osm";
    FILE* file = fopen(filename, "w");
    if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
    int num_sources = 1000;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment!\n");
        fprintf(file, "%f %f %f %f %f %f %f %f\n",
                i/10.0, i/20.0, 0.0, 1.0, 2.0, 3.0, 200.0e6, -0.7);
    }
    fclose(file);

    oskar_SkyModelGlobal_d sky;
    TIMER_START
    oskar_sky_model_load_d(filename, &sky);
    TIMER_STOP("Loaded %i sources", sky.num_sources)

    // Cleanup.
    remove(filename);

    const double deg2rad = 0.0174532925199432957692;

    // Check the data loaded correctly.
    CPPUNIT_ASSERT_EQUAL(num_sources, (int)sky.num_sources);
    for (int i = 0; i < num_sources; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, sky.RA[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, sky.Dec[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sky.I[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sky.Q[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, sky.U[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, sky.V[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, sky.reference_freq[i], 1.0e-6);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.7, sky.spectral_index[i], 1.0e-6);
    }
}


void SkyModelTest::test_load_new()
{
    const char* filename = "temp_sources.osm";
    FILE* file = fopen(filename, "w");
    if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
    int num_sources = 1000;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment!\n");
        fprintf(file, "%f %f %f %f %f %f %f %f\n",
                i/10.0, i/20.0, 0.0, 1.0, 2.0, 3.0, 200.0e6, -0.7);
    }
    fclose(file);


    oskar_SkyModel* sky = new oskar_SkyModel(0, OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    int err = oskar_sky_model_load(filename, sky);
    CPPUNIT_ASSERT_EQUAL_MESSAGE("oskar_SkyModel_load failed", 0, err);

    printf("%i\n", sky->RA.n_elements());

    CPPUNIT_ASSERT_EQUAL(num_sources, sky->num_sources);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());


//    oskar_SkyModelGlobal_d sky;
//    TIMER_START
//    oskar_sky_model_load_d(filename, &sky);
//    TIMER_STOP("Loaded %i sources", sky.num_sources)
//
//    // Cleanup.
//    remove(filename);
//
//    const double deg2rad = 0.0174532925199432957692;
//
//    // Check the data loaded correctly.
//    CPPUNIT_ASSERT_EQUAL(num_sources, (int)sky.num_sources);
//    for (int i = 0; i < num_sources; ++i)
//    {
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, sky.RA[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, sky.Dec[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, sky.I[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, sky.Q[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, sky.U[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, sky.V[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, sky.reference_freq[i], 1.0e-6);
//        CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.7, sky.spectral_index[i], 1.0e-6);
//    }
}
