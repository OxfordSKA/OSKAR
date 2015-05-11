/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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

#ifndef OSKAR_SETTINGS_UTILITY_STRING_HPP_
#define OSKAR_SETTINGS_UTILITY_STRING_HPP_

/**
 * @file oskar_settings_utility_string.hpp
 */

#include <string>
#include <vector>

// Trim whitespace from each end of a string.
std::string oskar_settings_utility_string_trim(const std::string& s,
        const std::string& whitespace = " \t");

// Split a string by CSV, skipping CSV within quotes.
std::vector<std::string> oskar_settings_utility_string_get_type_params(
        const std::string& s);

// Convert string to an integer.
int oskar_settings_utility_string_to_int(const std::string& s, bool* ok = 0);

// Convert integer to a string
std::string oskar_settings_utility_int_to_string(int i);

// Convert a string to upper case
std::string oskar_settings_utility_string_to_upper(const std::string& s);

// Returns true if string s1 starts with the string s2
bool oskar_settings_utility_string_starts_with(const std::string& s1,
        const std::string& s2, bool case_senstive = false);

// Convert double to a string
std::string oskar_settings_utility_double_to_string(double d,
        int precision = -17);

// Convert string to double
double oskar_settings_utility_string_to_double(const std::string& s, bool *ok);

// Generate a formatted string
std::string oskar_format_string(const std::string fmt, ...);

#endif /* OSKAR_SETTINGS_UTILITY_STRING_HPP_ */
