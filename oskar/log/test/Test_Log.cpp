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
    //oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_MESSAGE);
    //oskar_log_set_keep_file(false);
    int max_depth = 3;
    oskar_log_set_term_priority(OSKAR_LOG_DEBUG);
    oskar_log_message('M', -1, "This is a message");

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message('E', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message('W', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message('M', i, "depth %i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_message('D', i, "depth %i", i);
}

TEST(Log, oskar_log_value)
{
    oskar_log_create(OSKAR_LOG_WARNING, OSKAR_LOG_MESSAGE);
    oskar_log_set_keep_file(false);
    oskar_log_set_value_width(30);
    int max_depth = 3;
    oskar_log_message('M', -1, "%s", "Hello");
    oskar_log_value('M', 0, "prefix", "%s", "value");

    for (int i = 0; i < max_depth; ++i)
        oskar_log_value('E', i, "depth", "%i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_value('W', i, "depth", "%i", i);

    for (int i = 0; i < max_depth; ++i)
        oskar_log_value('M', i, "depth", "%i", i);

    oskar_log_set_term_priority(OSKAR_LOG_DEBUG);
    for (int i = 0; i < max_depth; ++i)
        oskar_log_value('D', i, "depth", "%i", i);
    oskar_log_free();
}

TEST(Log, oskar_log_section)
{
    oskar_log_set_term_priority(OSKAR_LOG_DEBUG);
    oskar_log_section('E', "%s", "message");
    oskar_log_section('W', "%s", "message");
    oskar_log_section('M', "%s", "message");
    oskar_log_section('D', "%s", "message");
}

TEST(Log, oskar_log_line)
{
    /*oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_NONE);*/
    /*oskar_log_set_keep_file(false);*/
    oskar_log_set_term_priority(OSKAR_LOG_DEBUG);
    oskar_log_line('E', '>');
    oskar_log_line('W', '+');
    oskar_log_line('M', '-');
    oskar_log_line('D', '*');
}

TEST(Log, oskar_log_error)
{
    oskar_log_error("This is an error");
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
    /*oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_DEBUG);*/
    oskar_log_set_term_priority(OSKAR_LOG_DEBUG);
    oskar_log_warning("This is a warning");
    oskar_log_error("This is an error");
    oskar_log_section('M', "This is a section");
    oskar_log_section('W', "This is a warning section");
    oskar_log_section('D', "This is a debug section");
}
