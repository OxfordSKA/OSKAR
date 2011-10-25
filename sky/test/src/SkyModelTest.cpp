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
#include <cstdio>
#include <cstdlib>


void SkyModelTest::test_resize()
{
    // Resizing on the GPU in single precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(10, OSKAR_SINGLE, OSKAR_LOCATION_GPU);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        sky->resize(1);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        sky->resize(20);
        CPPUNIT_ASSERT_EQUAL(20, sky->num_sources);
    }
    // Resizing on the CPU in double precision
    {
        oskar_SkyModel* sky = new oskar_SkyModel(10, OSKAR_DOUBLE, OSKAR_LOCATION_CPU);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_DOUBLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(10, sky->num_sources);
        sky->resize(1);
        CPPUNIT_ASSERT_EQUAL(1, sky->num_sources);
        sky->resize(20);
        CPPUNIT_ASSERT_EQUAL(20, sky->num_sources);
    }
}

void SkyModelTest::test_set_source()
{
    // Construct a sky model on the GPU of zero size.
    oskar_SkyModel* sky = new oskar_SkyModel(0, OSKAR_SINGLE, OSKAR_LOCATION_GPU);
    CPPUNIT_ASSERT_EQUAL(0, sky->num_sources);

    // Try to set a source into the model - this should fail as the model is
    // still zero size.
    int error = sky->set_source(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 200.0e6, -0.7);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_ERR_INVALID_ARGUMENT, error);

    // Resize the model to 2 sources.
    sky->resize(2);
    CPPUNIT_ASSERT_EQUAL(2, sky->num_sources);

    // Set values of these 2 sources.
    error = sky->set_source(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 200.0e6, -0.7);
    error = sky->set_source(1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 250.0e6, -0.8);
    CPPUNIT_ASSERT_EQUAL(0, error);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky->location());

    // Copy back into temp. structure on the CPU to check the values were set
    // correctly.
    oskar_SkyModel sky_temp(sky, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky_temp.location());
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_temp.type());
    CPPUNIT_ASSERT_EQUAL(2, sky_temp.num_sources);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ((float*)sky_temp.RA.data)[0], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, ((float*)sky_temp.reference_freq.data)[0], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(4.5, ((float*)sky_temp.Q.data)[1], 1.0e-6);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.8, ((float*)sky_temp.spectral_index.data)[1], 1.0e-6);
}


void SkyModelTest::test_append()
{
    int sky1_num_sources = 2;
    oskar_SkyModel* sky1 = new oskar_SkyModel(sky1_num_sources,
            OSKAR_SINGLE, OSKAR_LOCATION_GPU);
    for (int i = 0; i < sky1_num_sources; ++i)
    {
        double value = (double)i;
        sky1->set_source(i, value, value, value, value, value, value, value, value);
    }
    int sky2_num_sorces = 3;
    oskar_SkyModel* sky2 = new oskar_SkyModel(sky2_num_sorces,
            OSKAR_SINGLE, OSKAR_LOCATION_CPU);
    for (int i = 0; i < sky2_num_sorces; ++i)
    {
        double value = (double)i + 0.5;
        sky2->set_source(i, value, value, value, value, value, value, value, value);
    }
    sky1->append(sky2);
    CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky1->location());
    oskar_SkyModel sky_temp(sky1, OSKAR_LOCATION_CPU);
    CPPUNIT_ASSERT_EQUAL(sky1_num_sources + sky2_num_sorces, sky_temp.num_sources);
    for (int i = 0; i < sky_temp.num_sources; ++i)
    {
        if (i < sky1_num_sources)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)i, ((float*)sky_temp.RA.data)[i],
                    1.0e-6);
        }
        else
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i - sky1_num_sources) + 0.5, ((float*)sky_temp.RA.data)[i], 1.0e-6);
        }
    }
}


void SkyModelTest::test_load()
{
    const double deg2rad = 0.0174532925199432957692;
    const char* filename = "temp_sources.osm";

    // Load sky file with all columns specified.
    {
        FILE* file = fopen(filename, "w");
        if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f %f %f %f %f %f %f %f\n",
                    i/10.0, i/20.0, 0.0, 1.0, 2.0, 3.0, 200.0e6, -0.7);
        }
        fclose(file);


        oskar_SkyModel* sky = new oskar_SkyModel(0, OSKAR_SINGLE, OSKAR_LOCATION_CPU);
        int err = oskar_sky_model_load(filename, sky);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Cleanup.
        remove(filename);

        CPPUNIT_ASSERT_EQUAL_MESSAGE("oskar_SkyModel_load failed", 0, err);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_CPU, sky->location());
        CPPUNIT_ASSERT_EQUAL(num_sources, num_sources);
        CPPUNIT_ASSERT_EQUAL(num_sources, sky->RA.n_elements());
        CPPUNIT_ASSERT_EQUAL(num_sources, sky->rel_l.n_elements());

        // Check the data loaded correctly.
        for (int i = 0; i < num_sources; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, ((float*)sky->RA.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, ((float*)sky->Dec.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky->I.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ((float*)sky->Q.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, ((float*)sky->U.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(3.0, ((float*)sky->V.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(200.0e6, ((float*)sky->reference_freq.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(-0.7, ((float*)sky->spectral_index.data)[i], 1.0e-6);
        }

        free(sky);
    }


    // Load sky file with with just RA, Dec and I specified.
    {
        FILE* file = fopen(filename, "w");
        if (file == NULL) CPPUNIT_FAIL("Unable to create test file");
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f, %f, %f\n", i/10.0, i/20.0, (float)i);
        }
        fclose(file);

        // Load the sky model onto the GPU.
        oskar_SkyModel* sky_gpu = new oskar_SkyModel(0, OSKAR_SINGLE,
                OSKAR_LOCATION_GPU);
        int err = oskar_sky_model_load(filename, sky_gpu);
        CPPUNIT_ASSERT_EQUAL(0, err);

        // Cleanup.
        remove(filename);

        CPPUNIT_ASSERT_EQUAL_MESSAGE("oskar_SkyModel_load failed", 0, err);
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SINGLE, sky_gpu->type());
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_LOCATION_GPU, sky_gpu->location());
        CPPUNIT_ASSERT_EQUAL(num_sources, num_sources);
        CPPUNIT_ASSERT_EQUAL(num_sources, sky_gpu->RA.n_elements());
        CPPUNIT_ASSERT_EQUAL(num_sources, sky_gpu->rel_l.n_elements());

        // Copy the sky model back to the CPU and free the GPU version.
        oskar_SkyModel sky_cpu(sky_gpu, OSKAR_LOCATION_CPU);
        free(sky_gpu);

        // Check the data is correct.
        for (int i = 0; i < num_sources; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/10.0 * deg2rad, ((float*)sky_cpu.RA.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(i/20.0 * deg2rad, ((float*)sky_cpu.Dec.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL((float)i, ((float*)sky_cpu.I.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.Q.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.U.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.V.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.reference_freq.data)[i], 1.0e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ((float*)sky_cpu.spectral_index.data)[i], 1.0e-6);
        }
    }
}
