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

#include <gtest/gtest.h>

#include "utility/oskar_string_to_array.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

#define NUM_DOUBLES 6
#define NUM_STRINGS 5

// Doubles.

TEST(string_to_array_d, numeric_and_non_numeric)
{
    // Test comma and space separated values with additional non-numeric fields.
    double list[NUM_DOUBLES];
    char line[] = "hello 1.0,2.0 3.0, there,4.0     5.0 6.0";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)6, filled);
    for (size_t i = 0; i < filled; ++i)
        EXPECT_DOUBLE_EQ((double)(i+1), list[i]);
}

TEST(string_to_array_d, empty_string)
{
    // Test empty string.
    double list[NUM_DOUBLES];
    char line[] = "";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_d, single_space)
{
    // Test single space.
    double list[NUM_DOUBLES];
    char line[] = " ";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_d, negative_integers)
{
    // Test negative integers.
    double list[NUM_DOUBLES];
    char line[] = "-4,-3,-2 -1 0";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)5, filled);
    for (int i = 0; i < (int)filled; ++i)
        EXPECT_DOUBLE_EQ((double)(i-4), list[i]);
}

TEST(string_to_array_d, non_matching_string)
{
    // Test non-matching string.
    double list[NUM_DOUBLES];
    char line[] = "nobody home";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_d, too_many_items)
{
    // Test too many items.
    double list[NUM_DOUBLES];
    char line[] = "0.1 0.2 0.3   ,  0.4 0.5 0.6 0.7 0.8 0.9 1.0";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)NUM_DOUBLES, filled);
    for (size_t i = 0; i < filled; ++i)
        EXPECT_DOUBLE_EQ((i+1)/10.0, list[i]);
}

TEST(string_to_array_d, single_item)
{
    // Test single item.
    char line[] = "   0.1 ";
    double par;
    size_t filled = oskar_string_to_array_d(line, 1, &par);
    ASSERT_EQ((size_t)1, filled);
    EXPECT_DOUBLE_EQ(0.1, par);
}

TEST(string_to_array_d, comment_line)
{
    // Test comment line.
    double list[NUM_DOUBLES];
    char line[] = "# This is a comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_d, comment_line_with_space)
{
    // Test comment line with preceding space.
    double list[NUM_DOUBLES];
    char line[] = " # This is another comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_d, comment_at_end)
{
    // Test line with comment at end.
    double list[NUM_DOUBLES];
    char line[] = " 1.0 1.1 1.2 1.3 # This is another comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_DOUBLES, list);
    ASSERT_EQ((size_t)4, filled);
    for (size_t i = 0; i < filled; ++i)
        EXPECT_DOUBLE_EQ((double)(i/10.0 + 1), list[i]);
}

// Strings.

TEST(string_to_array_s, empty_string)
{
    // Test empty string.
    char *list[NUM_STRINGS];
    char line[] = "";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_s, single_space)
{
    // Test single space.
    char *list[NUM_STRINGS];
    char line[] = " ";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_s, normal_line)
{
    // Test normal use case.
    char *list[NUM_STRINGS];
    char line[] = "*, *, 10, 20, AZEL";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)5, filled);
    EXPECT_STREQ("*",    list[0]);
    EXPECT_STREQ("*",    list[1]);
    EXPECT_STREQ("10",   list[2]);
    EXPECT_STREQ("20",   list[3]);
    EXPECT_STREQ("AZEL", list[4]);
    ASSERT_EQ(list[0][0], '*');
    ASSERT_EQ(list[1][0], '*');
    ASSERT_EQ(list[4][0], 'A');
}

TEST(string_to_array_s, comment_line)
{
    // Test comment line.
    char *list[NUM_STRINGS];
    char line[] = "# This is a comment.";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_s, comment_line_with_space)
{
    // Test comment line with preceding space.
    char *list[NUM_STRINGS];
    char line[] = " # This is another comment.";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)0, filled);
}

TEST(string_to_array_s, comment_at_end)
{
    // Test line with comment at end.
    char *list[NUM_STRINGS];
    char line[] = " 1.0 1.1 1.2 1.3 # This is another comment.";
    size_t filled = oskar_string_to_array_s(line, NUM_STRINGS, list);
    ASSERT_EQ((size_t)4, filled);
    EXPECT_STREQ("1.0", list[0]);
    EXPECT_STREQ("1.1", list[1]);
    EXPECT_STREQ("1.2", list[2]);
    EXPECT_STREQ("1.3", list[3]);
}

// Strings (realloc).

TEST(string_to_array_realloc_s, reuse_buffer)
{
    // Test lines with comments and blanks.
    char **list = 0;
    size_t n = 0, filled = 0;

    {
        char line[] = "# This is a comment.";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)0, filled);
        ASSERT_EQ((size_t)0, n);
    }

    {
        char line[] = " # This is another comment.";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)0, filled);
        ASSERT_EQ((size_t)0, n);
    }

    {
        char line[] = "1, *, 10, 20, AZEL";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)5, filled);
        ASSERT_EQ((size_t)5, n);
        EXPECT_STREQ("1",    list[0]);
        EXPECT_STREQ("*",    list[1]);
        EXPECT_STREQ("10",   list[2]);
        EXPECT_STREQ("20",   list[3]);
        EXPECT_STREQ("AZEL", list[4]);
        ASSERT_EQ(list[0][0], '1');
        ASSERT_EQ(list[1][0], '*');
        ASSERT_EQ(list[4][0], 'A');
    }

    {
        char line[] = " ";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)0, filled);
        ASSERT_EQ((size_t)5, n);
    }

    {
        char line[] = "2, 0, 3, 34.5, 67.8, RADEC";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)6, filled);
        ASSERT_EQ((size_t)6, n);
        EXPECT_STREQ("2",     list[0]);
        EXPECT_STREQ("0",     list[1]);
        EXPECT_STREQ("3",     list[2]);
        EXPECT_STREQ("34.5",  list[3]);
        EXPECT_STREQ("67.8",  list[4]);
        EXPECT_STREQ("RADEC", list[5]);
        ASSERT_EQ(list[0][0], '2');
        ASSERT_EQ(list[1][0], '0');
        ASSERT_EQ(list[2][0], '3');
        ASSERT_EQ(list[5][0], 'R');
    }

    {
        char line[] = "1, 2, 50, 60, AZEL # Another comment";
        filled = oskar_string_to_array_realloc_s(line, &n, &list);
        ASSERT_EQ((size_t)5, filled);
        ASSERT_EQ((size_t)6, n);
        EXPECT_STREQ("1",    list[0]);
        EXPECT_STREQ("2",    list[1]);
        EXPECT_STREQ("50",   list[2]);
        EXPECT_STREQ("60",   list[3]);
        EXPECT_STREQ("AZEL", list[4]);
        ASSERT_EQ(list[0][0], '1');
        ASSERT_EQ(list[1][0], '2');
        ASSERT_EQ(list[4][0], 'A');
    }

    // Free the list array.
    free(list);
}
