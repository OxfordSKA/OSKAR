/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "utility/oskar_string_trim.h"


TEST(string_trim, test_empty_string)
{
    char line[] = "";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("", trimmed);
}


TEST(string_trim, test_trim_whitespace)
{
    char line[] = "  hello there     ";
    char* trimmed = oskar_string_trim(line, 0, 0);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes)
{
    char line[] = "'hello there'";
    char* trimmed = oskar_string_trim(line, 1, 0);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_with_whitespace)
{
    char line[] = "  \"  hello there \"     ";
    char* trimmed = oskar_string_trim(line, 1, 0);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_brackets)
{
    char line[] = "      [hello there]   ";
    char* trimmed = oskar_string_trim(line, 0, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_ticks_and_brackets)
{
    char line[] = "  ` (hello there)`     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_and_brackets)
{
    char line[] = "  \"(hello there)\"     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_and_brackets_without_quotes)
{
    char line[] = "[hello there] ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_and_brackets_without_brackets)
{
    char line[] = " '   hello there ' ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_with_only_brackets)
{
    char line[] = "    [  hello there  ]   ";
    char* trimmed = oskar_string_trim(line, 1, 0);
    EXPECT_STREQ("[  hello there  ]", trimmed);
}


TEST(string_trim, test_trim_brackets_with_only_quotes)
{
    char line[] = "    ' hello there'   ";
    char* trimmed = oskar_string_trim(line, 0, 1);
    EXPECT_STREQ("' hello there'", trimmed);
}


TEST(string_trim, test_trim_quotes_and_brackets_with_whitespace)
{
    char line[] = "  \"  (  hello there )   \"     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_ticks_and_brackets_with_whitespace)
{
    char line[] = "  `  (  hello there )   `     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_quotes_and_nested_brackets_with_whitespace)
{
    char line[] = "  \"  { [ hello there  ]}   \"     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}


TEST(string_trim, test_trim_nested_quotes_and_nested_brackets_with_whitespace)
{
    char line[] = "  \"  ' [( [ 'hello there   '  ])     ]'  \"     ";
    char* trimmed = oskar_string_trim(line, 1, 1);
    EXPECT_STREQ("hello there", trimmed);
}
