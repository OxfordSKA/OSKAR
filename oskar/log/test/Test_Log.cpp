/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "log/oskar_log.h"
#include <cstdio>

TEST(Log, oskar_log_message)
{
    int max_depth = 3;
    oskar_Log* log = 0;
    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    oskar_log_message(log, 'M', -1, "This is a message");

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message(log, 'E', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message(log, 'W', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message(log, 'M', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message(log, 'D', i, "depth %i", i);
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
        oskar_log_value(log, 'E', i, "depth", "%i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_value(log, 'W', i, "depth", "%i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_value(log, 'M', i, "depth", "%i", i);

    oskar_log_set_term_priority(log, OSKAR_LOG_DEBUG);
    for (int i = 0; i < max_depth; ++i)
        oskar_log_value(log, 'D', i, "depth", "%i", i);
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
