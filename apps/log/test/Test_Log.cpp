/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_log.h>
#include <cstdio>

#if 1
TEST(Log, oskar_log_list)
{
    //oskar_Log* log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_MESSAGE);
    //oskar_log_set_keep_file(log, false);
    oskar_Log* log = 0;
    int max_depth = 3;
    char priority;

    priority = 'E';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list(log, priority, i, "depth %i", i);
    }

    priority = 'W';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list(log, priority, i, "depth %i", i);
    }

    priority = 'M';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list(log, priority, i, "depth %i", i);
    }

    priority = 'D';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list(log, priority, i, "depth %i", i);
    }

    if (log) oskar_log_free(log);
}
#endif

#if 1
TEST(Log, oskar_log_list_value)
{
//    oskar_Log* log = oskar_log_create(OSKAR_LOG_MESSAGE, OSKAR_LOG_MESSAGE);
//    oskar_log_set_keep_file(log, false);
//    oskar_log_set_value_width(log, 30);
    oskar_Log* log = 0;
    int max_depth = 3;
    char priority;

    priority = 'E';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list_value(log, priority, i, "depth", "%i", i);
    }

    priority = 'W';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list_value(log, priority, i, "depth", "%i", i);
    }

    priority = 'M';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list_value(log, priority, i, "depth", "%i", i);
    }

    priority = 'D';
    for (int i = 0; i < max_depth; ++i) {
        oskar_log_list_value(log, priority, i, "depth", "%i", i);
    }

    if (log) oskar_log_free(log);
}
#endif

TEST(Log, oskar_log_value)
{
    oskar_Log* log = 0;

    oskar_log_value(log, 'E', "prefix", "%s", "value");
    oskar_log_value(log, 'W', "prefix", "%s", "value");
    oskar_log_value(log, 'M', "prefix", "%s", "value");
    oskar_log_value(log, 'D', "prefix", "%s", "value");

    if (log) oskar_log_free(log);
}

TEST(Log, oskar_log_message)
{
    oskar_Log* log = 0;

//    oskar_log_message(log, 'E', "%s", "message");
//    oskar_log_message(log, 'W', "%s", "message");
//    oskar_log_message(log, 'M', "%s", "message");
//    oskar_log_message(log, 'D', "%s", "message");
    oskar_log_list(log, 'E', -1, "%s", "message");
    oskar_log_list(log, 'W', -1, "%s", "message");
    oskar_log_list(log, 'M', -1, "%s", "message");
    oskar_log_list(log, 'D', -1, "%s", "message");

    if (log) oskar_log_free(log);
}

TEST(Log, oskar_log_section)
{
    oskar_Log* log = 0;

    oskar_log_section(log, 'E', "%s", "message");
    oskar_log_section(log, 'W', "%s", "message");
    oskar_log_section(log, 'M', "%s", "message");
    oskar_log_section(log, 'D', "%s", "message");

    if (log) oskar_log_free(log);
}

TEST(Log, oskar_log_line)
{
    oskar_Log* log = 0;
    /*oskar_Log* log = oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_NONE);*/
    /*oskar_log_set_keep_file(log, false);*/

    oskar_log_line(log, 'E', '>');
    oskar_log_line(log, 'W', '+');
    oskar_log_line(log, 'M', '-');
    oskar_log_line(log, 'D', '*');

    if (log) oskar_log_free(log);
}

TEST(Log, oskar_log_error)
{
    oskar_Log* log = 0;

    oskar_log_error(log, "This is an error.");

    // This is also an error...
    oskar_log_line(log, 'E', ' ');
    oskar_log_list(log, 'E', -101, "== ERROR: %s", "This is an error.");
    oskar_log_list(log, 'E', -1, "");

    if (log) oskar_log_free(log);
}

TEST(Log, depth_symbol)
{
    int max_depth = 10;
    char symbols[3] = {'+', '-', '*'};
    for (int i = 0; i < max_depth; ++i) {
        printf("d:%i r:%i sym:%c\n", i, i%3, symbols[i%3]);
    }
}

#if 0
TEST(Log, create_free)
{
    oskar_Log* log = oskar_log_create();
    oskar_log_set_keep_file(log, 0);
    oskar_log_free(log);
}

TEST(Log, section)
{
    oskar_Log* log = oskar_log_create();
    oskar_log_set_keep_file(log, 0);
    oskar_log_section(log, "Section %i", 0);
    oskar_log_section(log, "Section %i", 1);
    oskar_log_free(log);
}

TEST(Log, message)
{
    oskar_Log* log = oskar_log_create();
    oskar_log_set_keep_file(log, 0);
    for (int depth = 0; depth < 5; ++depth) {
        oskar_log_message(log, depth, "depth = %i", depth);
    }
    oskar_log_free(log);
}

/* TODO overloading of oskar_log_create */
/* http://locklessinc.com/articles/overloading/ */
/* 0 arguments = OSKAR_LOG_MESSAGE level */
/* 1 argument = log level */
/* could also consider using this to set other log parameters such as
 * keeping the file...
 * TODO why do the default error and warning messages not take a format
 * string?
 *
 * TODO: Update log functions
 *
 *  oskar_log_message() // no changes
 *  oskar_log_debug()   // no changes
 *  oskar_log_error()   // update to have same interface as message and debug
 *  oskar_log_warning() // update to have same interface as message and debug
 *
 *  oskar_log_line()    // This prints a line... at what level should it be?
 *                      // it probably should take a level argument...
 *
 * new functions:
 *
 *  oskar_log_box()?
 *
 * ... could just have:
 *
 *   oskar_log(oskar_Log* log, int depth, int level, const char* format, ...)
 *   oskar_log_line(oskar_Log*, int level, char symbol)
 *   oskar_log_box(oskar_Log*, int level, char symbol, int align=(left,right,centre), const char* format, ... )
 *
 * and overload this using the pre-processor.
 *
 * Consider also having a more machine readable log output (in addition to the
 * existing file).
 * i.e. some CSV type format with columns:
 *   time, level,  name, message, depth? ...
 * e.g.
 *
 *   ...
 *   "2014-08-14 00:47:47", MSG,  SETTINGS, "this is the message", 0
 *   "2014-08-14 00:47:47", MSG,  SETTINGS, "some value = 2", 1
 *   "2014-08-14 00:47:50", WARN,  JONES_E, "this is the message", 0
 *   ...
 *
 * with the log name that can be set into the log structure.
 * (this will be very useful for debugging)
 *
 *
 *
 */
TEST(Log, message_file_only)
{
    oskar_Log* log = oskar_log_create(OSKAR_LOG_NONE, OSKAR_LOG_NONE);
    oskar_log_set_keep_file(log, 1);
    oskar_log_set_term_priority(log, OSKAR_LOG_ERROR);
    oskar_log_set_file_priority(log, OSKAR_LOG_DEBUG);
    // mmm how does this interact with output file i/o ...
    // I'm guessing this is based on the log file rather than the terminal ...
    oskar_log_error(log, "This is an error.");
    oskar_log_warning_line(log, "This is a warning.");
    oskar_log_message(log, 0, "This is a message.");
    oskar_log_debug(log, 0, "This is a debug message.");
    oskar_log_line(log, '-');
    oskar_log_free(log);
}

TEST(Log, null_structure)
{
    oskar_log_message(0, 0, "This is a message");
}

TEST(Log, verbosity_level_values)
{
    // Test which flags match the default logging level.
    {
        int level = OSKAR_LOG_MESSAGE;
        ASSERT_TRUE(level  >= OSKAR_LOG_ERROR);
        ASSERT_TRUE(level  >= OSKAR_LOG_WARN);
        ASSERT_TRUE(level  >= OSKAR_LOG_MESSAGE);
        ASSERT_FALSE(level >= OSKAR_LOG_DEBUG);
    }

    // Test which flags match the WARN logging level
    {
        int level = OSKAR_LOG_WARN;
        ASSERT_TRUE(level  >= OSKAR_LOG_ERROR);
        ASSERT_TRUE(level  >= OSKAR_LOG_WARN);
        ASSERT_FALSE(level >= OSKAR_LOG_MESSAGE);
        ASSERT_FALSE(level >= OSKAR_LOG_DEBUG);
    }
}
#endif
