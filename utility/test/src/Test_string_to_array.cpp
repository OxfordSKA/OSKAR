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

#include "utility/test/Test_string_to_array.h"
#include "utility/oskar_string_to_array.h"

#include <cstdio>
#include <cstdlib>

void Test_string_to_array::test_method()
{
    double list[6];
    int filled = 0, n = sizeof(list) / sizeof(double);

    // Test comma and space separated values with additional non-numeric fields.
    char test1[] = "hello 1.0,2.0 3.0, there,4.0     5.0 6.0";
    filled = oskar_string_to_array_d(test1, n, list);
    CPPUNIT_ASSERT_EQUAL(6, filled);
    for (int i = 0; i < filled; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i+1), list[i], 1e-10);

    // Test empty string.
    char test2[] = "";
    filled = oskar_string_to_array_d(test2, n, list);
    CPPUNIT_ASSERT_EQUAL(0, filled);

    // Test empty string.
    char test3[] = " ";
    filled = oskar_string_to_array_d(test3, n, list);
    CPPUNIT_ASSERT_EQUAL(0, filled);

    // Test negative integers.
    char test4[] = "-4,-3,-2 -1 0";
    filled = oskar_string_to_array_d(test4, n, list);
    CPPUNIT_ASSERT_EQUAL(5, filled);
    for (int i = 0; i < filled; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i-4), list[i], 1e-10);

    // Test non-matching string.
    char test5[] = "nobody home";
    filled = oskar_string_to_array_d(test5, n, list);
    CPPUNIT_ASSERT_EQUAL(0, filled);

    // Test too many items.
    char test6[] = "0.1 0.2 0.3   ,  0.4 0.5 0.6 0.7 0.8 0.9 1.0";
    filled = oskar_string_to_array_d(test6, n, list);
    CPPUNIT_ASSERT_EQUAL(n, filled);
    for (int i = 0; i < filled; ++i)
        CPPUNIT_ASSERT_DOUBLES_EQUAL((i+1)/10.0, list[i], 1e-10);

    // Test single item.
    char test7[] = "   0.1 ";
    double par;
    filled = oskar_string_to_array_d(test7, 1, &par);
    CPPUNIT_ASSERT_EQUAL(1, filled);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, par, 1e-10);
}
