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

#include "math/test/Test_find_closest_match.h"
#include "math/oskar_find_closest_match.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"

void Test_find_closest_match::test()
{
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int size = 10;
    oskar_Mem values(type, location, size, OSKAR_TRUE);

    double start = 0.0;
    double inc = 0.3;

    double* values_ = (double*)values.data;
    for (int i = 0; i < size; ++i)
    {
        values_[i] = start + inc * i;
    }

    //  0   1   2   3   4   5   6   7   8   9
    // 0.0 0.3 0.6 0.9 1.2 1.5 1.8 2.1 2.4 2.7

    int idx;
    double value = 0.7;
    const oskar_Mem* v = &values;
    int err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(2, idx);

    value = 0.749999;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(2, idx);

    value = 0.75;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(3, idx);

    value = 0.750001;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(3, idx);

    value = 100;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(9, idx);

    value = -100;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(0, idx);

    value = 0.3;
    err = oskar_find_closest_match(&idx, value, v);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    CPPUNIT_ASSERT_EQUAL(1, idx);
}
