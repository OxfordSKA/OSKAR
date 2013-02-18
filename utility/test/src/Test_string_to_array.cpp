/*
 * Copyright (c) 2011-2013, The University of Oxford
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
#include <cstring>

#define NUM_DOUBLES 6
#define NUM_STRINGS 5

void Test_string_to_array::test_method()
{
    // Test comma and space separated values with additional non-numeric fields.
    {
        double list[NUM_DOUBLES];
        char test[] = "hello 1.0,2.0 3.0, there,4.0     5.0 6.0";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(6, filled);
        for (int i = 0; i < filled; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i+1), list[i], 1e-10);
    }

    // Test empty string.
    {
        double list[NUM_DOUBLES];
        char test[] = "";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test empty string.
    {
        double list[NUM_DOUBLES];
        char test[] = " ";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test negative integers.
    {
        double list[NUM_DOUBLES];
        char test[] = "-4,-3,-2 -1 0";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(5, filled);
        for (int i = 0; i < filled; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i-4), list[i], 1e-10);
    }

    // Test non-matching string.
    {
        double list[NUM_DOUBLES];
        char test[] = "nobody home";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test too many items.
    {
        double list[NUM_DOUBLES];
        char test[] = "0.1 0.2 0.3   ,  0.4 0.5 0.6 0.7 0.8 0.9 1.0";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(NUM_DOUBLES, filled);
        for (int i = 0; i < filled; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL((i+1)/10.0, list[i], 1e-10);
    }

    // Test single item.
    {
        char test[] = "   0.1 ";
        double par;
        int filled = oskar_string_to_array_d(test, 1, &par);
        CPPUNIT_ASSERT_EQUAL(1, filled);
        CPPUNIT_ASSERT_DOUBLES_EQUAL(0.1, par, 1e-10);
    }

    // Test comment line.
    {
        double list[NUM_DOUBLES];
        char test[] = "# This is a comment.";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test comment line with preceding space.
    {
        double list[NUM_DOUBLES];
        char test[] = " # This is another comment.";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test line with comment at end.
    {
        double list[NUM_DOUBLES];
        char test[] = " 1.0 1.1 1.2 1.3 # This is another comment.";
        int filled = oskar_string_to_array_d(test, NUM_DOUBLES, list);
        CPPUNIT_ASSERT_EQUAL(4, filled);
        for (int i = 0; i < filled; ++i)
            CPPUNIT_ASSERT_DOUBLES_EQUAL((double)(i/10.0 + 1), list[i], 1e-10);
    }
}

void Test_string_to_array::test_strings()
{
    // Test empty string.
    {
        char *list[NUM_STRINGS];
        char test[] = " ";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test empty string.
    {
        char *list[NUM_STRINGS];
        char test[] = "";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test normal use case.
    {
        char *list[NUM_STRINGS];
        char test[] = "*, *, 10, 20, AZEL";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(5, filled);
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[0], "*"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[1], "*"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[2], "10"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[3], "20"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[4], "AZEL"));
        CPPUNIT_ASSERT_EQUAL(list[0][0], '*');
        CPPUNIT_ASSERT_EQUAL(list[1][0], '*');
        CPPUNIT_ASSERT_EQUAL(list[4][0], 'A');
    }

    // Test comment line.
    {
        char *list[NUM_STRINGS];
        char test[] = "# This is a comment.";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test comment line with preceding space.
    {
        char *list[NUM_STRINGS];
        char test[] = " # This is another comment.";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
    }

    // Test line with comment at end.
    {
        char *list[NUM_STRINGS];
        char test[] = " 1.0 1.1 1.2 1.3 # This is another comment.";
        int filled = oskar_string_to_array_s(test, NUM_STRINGS, list);
        CPPUNIT_ASSERT_EQUAL(4, filled);
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[0], "1.0"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[1], "1.1"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[2], "1.2"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[3], "1.3"));
    }
}

void Test_string_to_array::test_strings_realloc()
{
    // Test lines with comments and blanks.
    {
        char **list = 0;
        int n = 0, filled = 0;
        char test1[] = "# This is a comment.";
        char test2[] = " # This is another comment.";
        char test3[] = "1, *, 10, 20, AZEL";
        char test4[] = " ";
        char test5[] = "2, 0, 3, 34.5, 67.8, RADEC";
        char test6[] = "1, 2, 50, 60, AZEL # Another comment";

        filled = oskar_string_to_array_realloc_s(test1, &n, &list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
        CPPUNIT_ASSERT_EQUAL(0, n);

        filled = oskar_string_to_array_realloc_s(test2, &n, &list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
        CPPUNIT_ASSERT_EQUAL(0, n);

        filled = oskar_string_to_array_realloc_s(test3, &n, &list);
        CPPUNIT_ASSERT_EQUAL(5, filled);
        CPPUNIT_ASSERT_EQUAL(5, n);
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[0], "1"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[1], "*"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[2], "10"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[3], "20"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[4], "AZEL"));
        CPPUNIT_ASSERT_EQUAL(list[0][0], '1');
        CPPUNIT_ASSERT_EQUAL(list[1][0], '*');
        CPPUNIT_ASSERT_EQUAL(list[4][0], 'A');

        filled = oskar_string_to_array_realloc_s(test4, &n, &list);
        CPPUNIT_ASSERT_EQUAL(0, filled);
        CPPUNIT_ASSERT_EQUAL(5, n);

        filled = oskar_string_to_array_realloc_s(test5, &n, &list);
        CPPUNIT_ASSERT_EQUAL(6, filled);
        CPPUNIT_ASSERT_EQUAL(6, n);
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[0], "2"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[1], "0"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[2], "3"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[3], "34.5"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[4], "67.8"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[5], "RADEC"));
        CPPUNIT_ASSERT_EQUAL(list[0][0], '2');
        CPPUNIT_ASSERT_EQUAL(list[1][0], '0');
        CPPUNIT_ASSERT_EQUAL(list[2][0], '3');
        CPPUNIT_ASSERT_EQUAL(list[5][0], 'R');

        filled = oskar_string_to_array_realloc_s(test6, &n, &list);
        CPPUNIT_ASSERT_EQUAL(5, filled);
        CPPUNIT_ASSERT_EQUAL(6, n);
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[0], "1"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[1], "2"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[2], "50"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[3], "60"));
        CPPUNIT_ASSERT_EQUAL(0, strcmp(list[4], "AZEL"));
        CPPUNIT_ASSERT_EQUAL(list[0][0], '1');
        CPPUNIT_ASSERT_EQUAL(list[1][0], '2');
        CPPUNIT_ASSERT_EQUAL(list[4][0], 'A');

        // Free the list array.
        free(list);
    }
}
