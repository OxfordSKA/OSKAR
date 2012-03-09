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

#include "math/test/Test_rotate.h"
#include "math/oskar_rotate.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"

#include <math.h>


void Test_rotate::test()
{
    int n = 1;
    oskar_Mem x(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n);
    oskar_Mem y(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n);
    oskar_Mem z(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n);

    double angle = 20.0 * M_PI/180.0;

    ((double*)x.data)[0] = 0.5;
    ((double*)y.data)[0] = 0.2;
    ((double*)z.data)[0] = 0.3;

    int err = oskar_rotate_x(1, &x, &y, &z, angle);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
    err = oskar_rotate_y(1, &x, &y, &z, angle);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
    err = oskar_rotate_z(1, &x, &y, &z, angle);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
}
