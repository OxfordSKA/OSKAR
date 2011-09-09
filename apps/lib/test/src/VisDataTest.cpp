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

#include "apps/lib/test/VisDataTest.h"
#include "interferometry/oskar_VisData.h"

#include <QtCore/QFile>
#include <cstdio>

void VisDataTest::test_load()
{
    printf("(%f)\n", CUDA_ARCH);

    const unsigned num_stations  = 25;
    const unsigned num_vis_dumps = 100;
    oskar_VisData_d data_in;
    const unsigned num_baselines = num_stations * (num_stations - 1) /2;
    oskar_allocate_vis_data_d(num_baselines * num_vis_dumps, &data_in);

    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        for (unsigned i = 0; i < num_baselines; ++i)
        {
            const unsigned index = j * num_baselines + i;
            data_in.u[index]     = (double)j;
            data_in.v[index]     = (double)j + 1.0;
            data_in.w[index]     = (double)j + 2.0;
            data_in.amp[index].x = (double)1.0;
            data_in.amp[index].y = (double)1.5;
        }
    }

    const char* filename = "temp_vis_data.dat";
    oskar_write_vis_data_d(filename, &data_in);

    oskar_VisData_d data_out;
    oskar_load_vis_data_d(filename, &data_out);

    CPPUNIT_ASSERT_EQUAL(data_in.num_samples, data_out.num_samples);

    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        for (unsigned i = 0; i < num_baselines; ++i)
        {
            const unsigned index = j * num_baselines + i;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(data_in.u[index], data_out.u[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(data_in.v[index], data_out.v[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(data_in.w[index], data_out.w[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(data_in.amp[index].x, data_out.amp[index].x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(data_in.amp[index].y, data_out.amp[index].y, 1e-6);
        }
    }

    oskar_free_vis_data_d(&data_in);
    oskar_free_vis_data_d(&data_out);

    QFile::remove(QString(filename));
}

