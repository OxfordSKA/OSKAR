/*
 * Copyright (c) 2011-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "utility/oskar_string_to_array.h"

#define NUM_VALUES 6

// Integers.

TEST(string_to_array_int, empty_string)
{
    int list[NUM_VALUES];
    char line[] = "";
    size_t filled = oskar_string_to_array_i(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_int, single_space)
{
    int list[NUM_VALUES];
    char line[] = " ";
    size_t filled = oskar_string_to_array_i(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_int, negative_integers)
{
    int list[NUM_VALUES];
    char line[] = "-4,-3,-2 -1 0";
    size_t filled = oskar_string_to_array_i(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 5, filled);
    for (int i = 0; i < (int) filled; ++i)
    {
        EXPECT_EQ(i - 4, list[i]);
    }
}


TEST(string_to_array_int, comment_at_end)
{
    int list[NUM_VALUES];
    char line[] = " 11 12 13 # This is a comment.";
    size_t filled = oskar_string_to_array_i(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 3, filled);
    for (int i = 0; i < (int) filled; ++i)
    {
        EXPECT_EQ(i + 11, list[i]);
    }
}


// Doubles.

TEST(string_to_array_d, numeric_and_non_numeric)
{
    // Test comma and space separated values with additional non-numeric fields.
    double list[NUM_VALUES];
    char line[] = "hello 1,2 3.0, there,4.0     5.0 6.0";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 6, filled);
    for (size_t i = 0; i < filled; ++i)
    {
        EXPECT_DOUBLE_EQ((double)(i+1), list[i]);
    }
}


TEST(string_to_array_d, empty_string)
{
    double list[NUM_VALUES];
    char line[] = "";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_d, single_space)
{
    double list[NUM_VALUES];
    char line[] = " ";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_d, negative_integers)
{
    double list[NUM_VALUES];
    char line[] = "-4,-3,-2 -1 0";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 5, filled);
    for (int i = 0; i < (int)filled; ++i)
    {
        EXPECT_DOUBLE_EQ((double)(i-4), list[i]);
    }
}


TEST(string_to_array_d, non_matching_string)
{
    double list[NUM_VALUES];
    char line[] = "nobody home";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t)0, filled);
}


TEST(string_to_array_d, too_many_items)
{
    double list[NUM_VALUES];
    char line[] = "0.1 0.2 0.3   ,  0.4 0.5 0.6 0.7 0.8 0.9 1.0";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) NUM_VALUES, filled);
    for (size_t i = 0; i < filled; ++i)
    {
        EXPECT_DOUBLE_EQ((i+1)/10.0, list[i]);
    }
}


TEST(string_to_array_d, single_item)
{
    char line[] = "   0.1 ";
    double par = 0.0;
    size_t filled = oskar_string_to_array_d(line, 1, &par);
    ASSERT_EQ((size_t) 1, filled);
    EXPECT_DOUBLE_EQ(0.1, par);
}


TEST(string_to_array_d, comment_line)
{
    double list[NUM_VALUES];
    char line[] = "# This is a comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_d, comment_line_with_space)
{
    double list[NUM_VALUES];
    char line[] = " # This is another comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 0, filled);
}


TEST(string_to_array_d, comment_at_end)
{
    double list[NUM_VALUES];
    char line[] = " 1.0 1.1 1.2 1.3 # This is another comment.";
    size_t filled = oskar_string_to_array_d(line, NUM_VALUES, list);
    ASSERT_EQ((size_t) 4, filled);
    for (size_t i = 0; i < filled; ++i)
    {
        EXPECT_DOUBLE_EQ((double)(i/10.0 + 1), list[i]);
    }
}
