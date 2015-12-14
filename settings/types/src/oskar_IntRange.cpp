/*
 * Copyright (c) 2015, The University of Oxford
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

#include <oskar_settings_utility_string.hpp>

#include <climits>
#include <vector>

#include <iostream>
#include <oskar_IntRange.hpp>

namespace oskar {

IntRange::IntRange() : min_(-INT_MAX), max_(INT_MAX), value_(0)
{
}

IntRange::~IntRange()
{
}

void IntRange::init(const std::string& s, bool* ok)
{
    if (*ok) *ok = true;

    min_ = -INT_MAX;
    max_ =  INT_MAX;

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -INT_MAX to INT_MAX)
    //  - With 1 entry the range is (p[0] to INT_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    //
    // Notes: if p[0] is the string 'MIN' or p[1] is the string 'MAX'
    // these will resolve as -INT_MAX and INT_MAX respectively.
    //
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u) {
        return;
    }
    else if (p.size() == 1u) {
        if (p[0] == "MIN") min_ = -INT_MAX;
        else min_ = oskar_settings_utility_string_to_int(p[0], ok);
        return;
    }
    else if (p.size() == 2u) {
        if (p[0] == "MIN") min_ = -INT_MAX;
        else min_ = oskar_settings_utility_string_to_int(p[0], ok);
        if (ok && !*ok) return;
        if (p[1] == "MAX") max_ = INT_MAX;
        else max_ = oskar_settings_utility_string_to_int(p[1], ok);
        return;
    }

    // If more than 3 parameters, set the status to false.
    if (*ok) *ok = false;
}

void IntRange::fromString(const std::string& s, bool* ok)
{
    if (ok) *ok = false;

    int i = oskar_settings_utility_string_to_int(s, ok);
    if (ok && !*ok) return;

    if (i >= min_ && i <= max_) {
        if (ok) *ok = true;
        value_ = i;
    }
    else if (i < min_) { if (ok) *ok = false; value_ = min_; }
    else if (i > max_) { if (ok) *ok = false; value_ = max_; }
}

std::string IntRange::toString() const
{
    return oskar_settings_utility_int_to_string(value_);
}

} // namespace oskar

