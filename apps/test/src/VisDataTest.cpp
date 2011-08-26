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

#include "apps/test/VisDataTest.h"
#include "apps/oskar_VisData.h"

#include <QtCore/QFile>
#include <cstdio>

void VisDataTest::test_load()
{
    const unsigned num_stations  = 25;
    const unsigned num_vis_dumps = 100;
    oskar_VisData data_in(num_stations, num_vis_dumps);
    const unsigned num_baselines = data_in.num_baselines();

    double* u_in    = data_in.u();
    double* v_in    = data_in.v();
    double* w_in    = data_in.w();
    double2* vis_in = data_in.vis();

    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        for (unsigned i = 0; i < num_baselines; ++i)
        {
            const unsigned index = j * num_baselines + i;
            u_in[index]     = (double)j;
            v_in[index]     = (double)j + 1.0;
            w_in[index]     = (double)j + 2.0;
            vis_in[index].x = (double)1.0;
            vis_in[index].y = (double)1.5;
        }
    }

    const char* filename = "temp_vis_data.dat";
    data_in.write(filename);

    oskar_VisData data_out(0, 0);
    data_out.load(filename);

    CPPUNIT_ASSERT_EQUAL(data_in.size(), data_out.size());

    double* u_out    = data_out.u();
    double* v_out    = data_out.v();
    double* w_out    = data_out.w();
    double2* vis_out = data_out.vis();

    for (unsigned j = 0; j < num_vis_dumps; ++j)
    {
        for (unsigned i = 0; i < num_baselines; ++i)
        {
            const unsigned index = j * num_baselines + i;
            CPPUNIT_ASSERT_DOUBLES_EQUAL(u_in[index], u_out[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(v_in[index], v_out[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(w_in[index], w_out[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(w_in[index], w_out[index], 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(vis_in[index].x, vis_out[index].x, 1e-6);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(vis_in[index].y, vis_out[index].y, 1e-6);
        }
    }

    QFile::remove(QString(filename));
}

