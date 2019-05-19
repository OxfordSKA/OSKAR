/*
 * Copyright (c) 2012-2019, The University of Oxford
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

#ifndef OSKAR_OPTION_PARSER_H_
#define OSKAR_OPTION_PARSER_H_

/**
 * @file oskar_option_parser.h
 */

#include <settings/oskar_settings_macros.h>

namespace oskar {

struct OptionParserPrivate;

/**
 * @brief
 * Provides a command line parser for OSKAR applications.
 *
 * @details
 * Provides a command line parser for OSKAR applications.
 *
 * Note on the use symbols in option syntax: (the following are advised
 * in order to maintain consistency)
 *
 *  []   = optional (e.g. $ foo [settings file] )
 *  <>   = required (e.g. $ foo <file name> )
 *  ...  = repeating elements, "and so on"
 *  |    = mutually exclusive
 *
 * TODO(BM) Better handling of unexpected options.
 * It would be useful if a warning could be printed.
 */
class OptionParser
{
public:
    OSKAR_SETTINGS_EXPORT
    OptionParser(const char* title, const char* ver,
            const char* settings = "");

    OSKAR_SETTINGS_EXPORT
    virtual ~OptionParser();

    OSKAR_SETTINGS_EXPORT
    void add_example(const char* text);

    // Wrapper to define flags with no arguments
    OSKAR_SETTINGS_EXPORT
    void add_flag(const char* flag1, const char* help, bool required = false,
            const char* flag2 = 0);

    // Wrapper to define flags with arguments with default values.
    OSKAR_SETTINGS_EXPORT
    void add_flag(const char* flag1, const char* help, int expected_args,
            const char* defaults = "", bool required = false,
            const char* flag2 = 0);

    OSKAR_SETTINGS_EXPORT
    void add_optional(const char* name, const char* help = "");

    OSKAR_SETTINGS_EXPORT
    void add_required(const char* name, const char* help = "");

    OSKAR_SETTINGS_EXPORT
    void add_settings_options();

    OSKAR_SETTINGS_EXPORT
    bool check_options(int argc, char** argv);

    OSKAR_SETTINGS_EXPORT
    void error(const char* format, ...);

    OSKAR_SETTINGS_EXPORT
    const char* get_arg(int i = 0) const;

    OSKAR_SETTINGS_EXPORT
    double get_double(const char* name);

    OSKAR_SETTINGS_EXPORT
    int get_int(const char* name);

    OSKAR_SETTINGS_EXPORT
    const char* get_string(const char* name);

    OSKAR_SETTINGS_EXPORT
    const char* const* get_input_files(int min_required, int* num_files);

    OSKAR_SETTINGS_EXPORT
    int is_set(const char* option);

    OSKAR_SETTINGS_EXPORT
    void print_usage();

    OSKAR_SETTINGS_EXPORT
    int num_args() const;

    OSKAR_SETTINGS_EXPORT
    void set_description(const char* description);

    OSKAR_SETTINGS_EXPORT
    void set_settings(const char* text);

    OSKAR_SETTINGS_EXPORT
    void set_title(const char* text);

    OSKAR_SETTINGS_EXPORT
    void set_version(const char* version, bool show = true);

private:
    OptionParserPrivate* p;
};

} /* namespace oskar */

#endif /* OSKAR_OPTION_PARSER_H_ */
