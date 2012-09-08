/*
 * Copyright (c) 2012, The University of Oxford
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

#include "interferometry/test/Test_evaluate_baselines.h"
#include "interferometry/oskar_evaluate_baselines.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"

/**
 * @details
 * Tests baseline evaluation for 3 stations.
 */
void Test_evaluate_baselines::test_small()
{
    int status = 0;
    int num_stations = 3;
    int num_baselines = num_stations * (num_stations - 1) / 2;

    // Allocate host memory.
    oskar_Mem u(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem v(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem w(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem uu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);
    oskar_Mem vv(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);
    oskar_Mem ww(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);

    // Fill station coordinates with test data.
    for (int i = 0; i < num_stations; ++i)
    {
        ((double*)u)[i] = (double)(i + 1);
        ((double*)v)[i] = (double)(i + 2);
        ((double*)w)[i] = (double)(i + 3);
    }

    // Evaluate baseline coordinates on CPU.
    oskar_evaluate_baselines(&uu, &vv, &ww, &u, &v, &w, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    // Check results are correct.
    for (int s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (int s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)uu)[b],
                    ((double*)u)[s2] - ((double*)u)[s1], 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)vv)[b],
                    ((double*)v)[s2] - ((double*)v)[s1], 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)ww)[b],
                    ((double*)w)[s2] - ((double*)w)[s1], 1e-10);
        }
    }

    // Allocate device memory and copy input data.
    oskar_Mem u_gpu(&u, OSKAR_LOCATION_GPU);
    oskar_Mem v_gpu(&v, OSKAR_LOCATION_GPU);
    oskar_Mem w_gpu(&w, OSKAR_LOCATION_GPU);
    oskar_Mem uu_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);
    oskar_Mem vv_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);
    oskar_Mem ww_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);

    // Evaluate baseline coordinates on GPU.
    oskar_evaluate_baselines(&uu_gpu, &vv_gpu, &ww_gpu, &u_gpu, &v_gpu, &w_gpu,
            &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    // Copy results back.
    oskar_Mem uu2(&uu_gpu, OSKAR_LOCATION_CPU);
    oskar_Mem vv2(&vv_gpu, OSKAR_LOCATION_CPU);
    oskar_Mem ww2(&ww_gpu, OSKAR_LOCATION_CPU);

    // Check results are correct.
    for (int i = 0; i < num_baselines; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)uu)[i], ((double*)uu2)[i],
                1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)vv)[i], ((double*)vv2)[i],
                1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)ww)[i], ((double*)ww2)[i],
                1e-10);
    }
}

/**
 * @details
 * Tests baseline evaluation for 50 stations.
 */
void Test_evaluate_baselines::test_large()
{
    int status = 0;
    int num_stations = 50;
    int num_baselines = num_stations * (num_stations - 1) / 2;

    // Allocate host memory.
    oskar_Mem u(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem v(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem w(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_stations);
    oskar_Mem uu(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);
    oskar_Mem vv(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);
    oskar_Mem ww(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_baselines);

    // Fill station coordinates with test data.
    for (int i = 0; i < num_stations; ++i)
    {
        ((double*)u)[i] = (double)(i + 1);
        ((double*)v)[i] = (double)(i + 2);
        ((double*)w)[i] = (double)(i + 3);
    }

    // Evaluate baseline coordinates on CPU.
    oskar_evaluate_baselines(&uu, &vv, &ww, &u, &v, &w, &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    // Check results are correct.
    for (int s1 = 0, b = 0; s1 < num_stations; ++s1)
    {
        for (int s2 = s1 + 1; s2 < num_stations; ++s2, ++b)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)uu)[b],
                    ((double*)u)[s2] - ((double*)u)[s1], 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)vv)[b],
                    ((double*)v)[s2] - ((double*)v)[s1], 1e-10);
            CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)ww)[b],
                    ((double*)w)[s2] - ((double*)w)[s1], 1e-10);
        }
    }

    // Allocate device memory and copy input data.
    oskar_Mem u_gpu(&u, OSKAR_LOCATION_GPU);
    oskar_Mem v_gpu(&v, OSKAR_LOCATION_GPU);
    oskar_Mem w_gpu(&w, OSKAR_LOCATION_GPU);
    oskar_Mem uu_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);
    oskar_Mem vv_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);
    oskar_Mem ww_gpu(OSKAR_DOUBLE, OSKAR_LOCATION_GPU, num_baselines);

    // Evaluate baseline coordinates on GPU.
    oskar_evaluate_baselines(&uu_gpu, &vv_gpu, &ww_gpu, &u_gpu, &v_gpu, &w_gpu,
            &status);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(status), 0, status);

    // Copy results back.
    oskar_Mem uu2(&uu_gpu, OSKAR_LOCATION_CPU);
    oskar_Mem vv2(&vv_gpu, OSKAR_LOCATION_CPU);
    oskar_Mem ww2(&ww_gpu, OSKAR_LOCATION_CPU);

    // Check results are correct.
    for (int i = 0; i < num_baselines; ++i)
    {
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)uu)[i], ((double*)uu2)[i],
                1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)vv)[i], ((double*)vv2)[i],
                1e-10);
        CPPUNIT_ASSERT_DOUBLES_EQUAL( ((double*)ww)[i], ((double*)ww2)[i],
                1e-10);
    }
}
