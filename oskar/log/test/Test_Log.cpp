/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "log/oskar_log.h"
#include <cstdio>

TEST(Log, oskar_log_message)
{
    int max_depth = 3;
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    oskar_log_message(log, 'M', -1, "This is a message");

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_message(log, 'E', i, "depth %i", i);
    }

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_message(log, 'W', i, "depth %i", i);
    }

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_message(log, 'M', i, "depth %i", i);
    }

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_message(log, 'D', i, "depth %i", i);
    }
}

TEST(Log, oskar_log_value)
{
    oskar_Log* log = 0;
    oskar_log_set_file_priority(log, OSKAR_LOG_WARNING);
    oskar_log_set_term_priority(log, OSKAR_LOG_MESSAGE);
    oskar_log_set_keep_file(log, false);
    oskar_log_set_value_width(log, 30);
    int max_depth = 3;
    oskar_log_message(log, 'M', -1, "%s", "Hello");
    oskar_log_value(log, 'M', 0, "prefix", "%s", "value");

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_value(log, 'E', i, "depth", "%i", i);
    }

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_value(log, 'W', i, "depth", "%i", i);
    }

    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_value(log, 'M', i, "depth", "%i", i);
    }

    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    for (int i = 0; i < max_depth; ++i)
    {
        oskar_log_value(log, 'D', i, "depth", "%i", i);
    }
    oskar_log_free(log);
}

TEST(Log, oskar_log_section)
{
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    oskar_log_section(log, 'E', "%s", "message");
    oskar_log_section(log, 'W', "%s", "message");
    oskar_log_section(log, 'M', "%s", "message");
    oskar_log_section(log, 'D', "%s", "message");
}

TEST(Log, oskar_log_line)
{
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    oskar_log_line(log, 'E', '>');
    oskar_log_line(log, 'W', '+');
    oskar_log_line(log, 'M', '-');
    oskar_log_line(log, 'D', '*');
}

TEST(Log, oskar_log_error)
{
    oskar_Log* log = 0;
    oskar_log_error(log, "This is an error");
}

TEST(Log, depth_symbol)
{
    int max_depth = 10;
    char symbols[3] = {'+', '-', '*'};
    for (int i = 0; i < max_depth; ++i) {
        printf("d:%i r:%i sym:%c\n", i, i%3, symbols[i%3]);
    }
}

TEST(Log, special_methods)
{
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    oskar_log_warning(log, "This is a warning");
    oskar_log_error(log, "This is an error");
    oskar_log_section(log, 'M', "This is a section");
    oskar_log_section(log, 'W', "This is a warning section");
    oskar_log_section(log, 'D', "This is a debug section");
}
