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

#include "math/test/Test_fit_ellipse.h"
#include "math/oskar_fit_ellipse.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_Mem.h"

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

void Test_fit_ellipse::test()
{
    double maj = 0.0, min = 0.0, pa = 0.0;

    {
        int num_points = 7;
        oskar_Mem x(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);
        oskar_Mem y(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);
        ((double*)x.data)[0] = -0.1686;
        ((double*)x.data)[1] = -0.0921;
        ((double*)x.data)[2] =  0.0765;
        ((double*)x.data)[3] =  0.1686;
        ((double*)x.data)[4] =  0.0921;
        ((double*)x.data)[5] = -0.0765;
        ((double*)x.data)[6] = -0.1686;

        ((double*)y.data)[0] =  0.7282;
        ((double*)y.data)[1] =  0.6994;
        ((double*)y.data)[2] =  0.6675;
        ((double*)y.data)[3] =  0.6643;
        ((double*)y.data)[4] =  0.7088;
        ((double*)y.data)[5] =  0.7407;
        ((double*)y.data)[6] =  0.7282;

        int err = oskar_fit_ellipse(&maj, &min, &pa, num_points, &x, &y);
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);

        printf("\n%f %f %f\n", maj*180.0/M_PI, min*180.0/M_PI, pa*180.0/M_PI);
    }

//    {
//        int num_points = 6;
//        oskar_Mem x(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);
//        oskar_Mem y(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, num_points, OSKAR_TRUE);
//        ((double*)x.data)[0] = 1;
//        ((double*)x.data)[1] = 2;
//        ((double*)x.data)[2] = 3;
//        ((double*)x.data)[3] = 4;
//        ((double*)x.data)[4] = 5;
//        ((double*)x.data)[5] = 6;
//
//        ((double*)y.data)[0] = 1;
//        ((double*)y.data)[1] = 2;
//        ((double*)y.data)[2] = 3;
//        ((double*)y.data)[3] = 4;
//        ((double*)y.data)[4] = 5;
//        ((double*)y.data)[5] = 6;
//
//        int err = oskar_fit_ellipse(&maj, &min, &pa, num_points, &x, &y);
//        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
//
//        printf("\n%f %f %f\n", maj*180.0/M_PI, min*180.0/M_PI, pa*180.0/M_PI);
//    }
}
