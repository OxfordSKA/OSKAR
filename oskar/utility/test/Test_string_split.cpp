/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gtest/gtest.h>

#include "utility/oskar_string_split.h"


TEST(string_split, test_reuse_list_buffer)
{
    int status = 0;
    int num_found = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    {
        char line[] = "# This is a comment.";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        EXPECT_EQ(0, num_found);
        ASSERT_EQ(0, list_size);
    }
    {
        char line[] = " # This is another comment.";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        EXPECT_EQ(0, num_found);
        ASSERT_EQ(0, list_size);
    }
    {
        char line[] = "1, *, 10, 20, AZEL";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        EXPECT_EQ(5, num_found);
        ASSERT_EQ(5, list_size);
        EXPECT_STREQ("1", list[0]);
        EXPECT_STREQ("*", list[1]);
        EXPECT_STREQ("10", list[2]);
        EXPECT_STREQ("20", list[3]);
        EXPECT_STREQ("AZEL", list[4]);
    }
    {
        char line[] = " ";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        ASSERT_EQ(0, num_found);
        ASSERT_EQ(5, list_size);
    }
    {
        char line[] = "2, 0, 3, 34.5, 67.8, RADEC";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        EXPECT_EQ(6, num_found);
        ASSERT_EQ(6, list_size);
        EXPECT_STREQ("2", list[0]);
        EXPECT_STREQ("0", list[1]);
        EXPECT_STREQ("3", list[2]);
        EXPECT_STREQ("34.5", list[3]);
        EXPECT_STREQ("67.8", list[4]);
        EXPECT_STREQ("RADEC", list[5]);
    }
    {
        char line[] = "1, 2, 50, 60, AZEL # Another comment";
        num_found = oskar_string_split(
                line, &list_size, &list, split_on_equals, &status
        );
        ASSERT_EQ(0, status);
        EXPECT_EQ(5, num_found);
        ASSERT_EQ(6, list_size);
        EXPECT_STREQ("1", list[0]);
        EXPECT_STREQ("2", list[1]);
        EXPECT_STREQ("50", list[2]);
        EXPECT_STREQ("60", list[3]);
        EXPECT_STREQ("AZEL", list[4]);
    }
    free(list);
}


TEST(string_split, test_basic_whitespace)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "alpha beta gamma";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    EXPECT_STREQ("alpha", list[0]);
    EXPECT_STREQ("beta", list[1]);
    EXPECT_STREQ("gamma", list[2]);
    free(list);
}


TEST(string_split, test_trailing_whitespace)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "alpha   beta\tgamma  ";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    EXPECT_STREQ("alpha", list[0]);
    EXPECT_STREQ("beta", list[1]);
    EXPECT_STREQ("gamma", list[2]);
    free(list);
}


TEST(string_split, test_comma_split)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "alpha ,beta,gamma , delta";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(4, num_found);
    ASSERT_EQ(4, list_size);
    EXPECT_STREQ("alpha", list[0]);
    EXPECT_STREQ("beta", list[1]);
    EXPECT_STREQ("gamma", list[2]);
    EXPECT_STREQ("delta", list[3]);
    free(list);
}


TEST(string_split, test_mixed_split)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "alpha,beta gamma    ,          delta   ";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(4, num_found);
    ASSERT_EQ(4, list_size);
    EXPECT_STREQ("alpha", list[0]);
    EXPECT_STREQ("beta", list[1]);
    EXPECT_STREQ("gamma", list[2]);
    EXPECT_STREQ("delta", list[3]);
    free(list);
}


TEST(string_split, test_comment)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "alpha beta # a comment";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("alpha", list[0]);
    EXPECT_STREQ("beta", list[1]);
    free(list);
}


TEST(string_split, test_values_and_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "  1.1, 2.2 [0.1, 0.2, 0.3] , (0.4 0.5)   [0.6,0.7]  ";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(5, num_found);
    ASSERT_EQ(5, list_size);
    EXPECT_STREQ("1.1", list[0]);
    EXPECT_STREQ("2.2", list[1]);
    EXPECT_STREQ("[0.1, 0.2, 0.3]", list[2]);
    EXPECT_STREQ("(0.4 0.5)", list[3]);
    EXPECT_STREQ("[0.6,0.7]", list[4]);
    free(list);
}


TEST(string_split, test_many_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "(a, [b, {c, d}]) [e f g]";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("(a, [b, {c, d}])", list[0]);
    EXPECT_STREQ("[e f g]", list[1]);
    free(list);
}


TEST(string_split, test_nested_brackets_same_type)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "(a, (b, c), d) next";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("(a, (b, c), d)", list[0]);
    EXPECT_STREQ("next", list[1]);
    free(list);
}


TEST(string_split, test_hash_inside_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "(#not a comment) next # a real comment";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("(#not a comment)", list[0]);
    EXPECT_STREQ("next", list[1]);
    free(list);
}


TEST(string_split, test_hash_inside_quotes)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "'#not a comment'     next # a real comment";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("'#not a comment'", list[0]);
    EXPECT_STREQ("next", list[1]);
    free(list);
}


TEST(string_split, test_quotes_override_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "' (1,2,3)  ', '[4,5,6]'";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("' (1,2,3)  '", list[0]);
    EXPECT_STREQ("'[4,5,6]'", list[1]);
    free(list);
}


TEST(string_split, test_quotes_override_missing_closing_bracket)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "\"(1,2,3\", [4,5,6], (7  8  9)";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    EXPECT_STREQ("\"(1,2,3\"", list[0]);
    EXPECT_STREQ("[4,5,6]", list[1]);
    EXPECT_STREQ("(7  8  9)", list[2]);
    free(list);
}


TEST(string_split, test_quoted_token_with_trailing_whitespace_then_comma)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "  \"quoted with extra spaces, then comma\"   ,  next_token";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("\"quoted with extra spaces, then comma\"", list[0]);
    EXPECT_STREQ("next_token", list[1]);
    free(list);
}


TEST(string_split, test_empty_fields_only)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = ",,,";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(4, num_found);
    ASSERT_EQ(4, list_size);
    EXPECT_STREQ("", list[0]);
    EXPECT_STREQ("", list[1]);
    EXPECT_STREQ("", list[2]);
    EXPECT_STREQ("", list[3]);
    free(list);
}


TEST(string_split, test_spaces_only)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = " ,    , , ";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(4, num_found);
    ASSERT_EQ(4, list_size);
    EXPECT_STREQ("", list[0]);
    EXPECT_STREQ("", list[1]);
    EXPECT_STREQ("", list[2]);
    EXPECT_STREQ("", list[3]);
    free(list);
}


TEST(string_split, test_leading_middle_trailing_empty_fields)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = " ,a,,b,";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(5, num_found);
    ASSERT_EQ(5, list_size);
    EXPECT_STREQ("", list[0]);
    EXPECT_STREQ("a", list[1]);
    EXPECT_STREQ("", list[2]);
    EXPECT_STREQ("b", list[3]);
    EXPECT_STREQ("", list[4]);
    free(list);
}


TEST(string_split, test_quoted_and_empty_fields)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "'(1,2,3)',,'[4,5,6]'";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    EXPECT_STREQ("'(1,2,3)'", list[0]);
    EXPECT_STREQ("", list[1]);
    EXPECT_STREQ("'[4,5,6]'", list[2]);
    free(list);
}


TEST(string_split, test_empty_before_nonempty_with_spaces)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = ",   next";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("", list[0]);
    EXPECT_STREQ("next", list[1]);
    free(list);
}


TEST(string_split, test_unclosed_quote)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "'unclosed quote";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    EXPECT_EQ(1, num_found);
    ASSERT_EQ(1, list_size);
    free(list);
}


TEST(string_split, test_unclosed_bracket)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "(0.1, 0.2 11.2 22.3";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    EXPECT_EQ(1, num_found);
    ASSERT_EQ(1, list_size);
    free(list);
}


TEST(string_split, test_non_printable)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "0.1, 0.2 \x03\x04 1.1 2.2";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    free(list);
}


TEST(string_split, test_non_printable_inside_quotes)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "0.1, 0.2 \"argh \x03\x04\" 1.1 2.2";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    free(list);
}


TEST(string_split, test_non_printable_inside_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "0.1, 0.2 [again \x03\x04] 1.1 2.2";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(OSKAR_ERR_INVALID_ARGUMENT, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    free(list);
}


TEST(string_split, test_empty_input_gives_no_tokens)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(0, num_found);
    ASSERT_EQ(0, list_size);
    free(list);
}


TEST(string_split, test_whitespace_only_gives_no_tokens)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "    \t   ";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(0, num_found);
    ASSERT_EQ(0, list_size);
    free(list);
}


TEST(string_split, test_comment_only_gives_no_tokens)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "# A comment";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(0, num_found);
    ASSERT_EQ(0, list_size);
    free(list);
}


TEST(string_split, test_whitespace_then_comment_gives_no_tokens)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "      # A comment";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(0, num_found);
    ASSERT_EQ(0, list_size);
    free(list);
}


TEST(string_split, test_equals_keeps_together)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 0;
    char** list = 0;
    char line[] = "Ra Dec I='10.0' Q= 2, U , V  =   0.5 SpectralIndex ='[1,2]'";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(7, num_found);
    ASSERT_EQ(7, list_size);
    EXPECT_STREQ("Ra", list[0]);
    EXPECT_STREQ("Dec", list[1]);
    EXPECT_STREQ("I='10.0'", list[2]);
    EXPECT_STREQ("Q= 2", list[3]);
    EXPECT_STREQ("U", list[4]);
    EXPECT_STREQ("V  =   0.5", list[5]);
    EXPECT_STREQ("SpectralIndex ='[1,2]'", list[6]);
    free(list);
}


TEST(string_split, test_equals_splits)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 1;
    char** list = 0;
    char line[] = "parameter_value=0.123";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("parameter_value", list[0]);
    EXPECT_STREQ("0.123", list[1]);
    free(list);
}


TEST(string_split, test_equals_splits_keeping_quotes)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 1;
    char** list = 0;
    char line[] = "parameter_value='0.123'";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("parameter_value", list[0]);
    EXPECT_STREQ("'0.123'", list[1]);
    free(list);
}


TEST(string_split, test_equals_splits_once_with_trailing_words)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 1;
    char** list = 0;
    char line[] = "parameter value =10.0 = and more = = =";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(3, num_found);
    ASSERT_EQ(3, list_size);
    EXPECT_STREQ("parameter", list[0]);
    EXPECT_STREQ("value", list[1]);
    EXPECT_STREQ("10.0 = and more = = =", list[2]);
    free(list);
}


TEST(string_split, test_equals_splits_only_outside_brackets)
{
    int status = 0;
    int list_size = 0;
    int split_on_equals = 1;
    char** list = 0;
    char line[] = "(Ra Dec I Q U V ReferenceFrequency = '100e6') = format";
    const int num_found = oskar_string_split(
            line, &list_size, &list, split_on_equals, &status
    );
    ASSERT_EQ(0, status);
    EXPECT_EQ(2, num_found);
    ASSERT_EQ(2, list_size);
    EXPECT_STREQ("(Ra Dec I Q U V ReferenceFrequency = '100e6')", list[0]);
    EXPECT_STREQ("format", list[1]);
    free(list);
}
