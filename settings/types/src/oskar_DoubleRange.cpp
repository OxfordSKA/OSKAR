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

#include <cfloat>
#include <vector>

#include <iostream>
#include <oskar_DoubleRange.hpp>

namespace oskar {

DoubleRange::DoubleRange() : min_(-DBL_MAX), max_(DBL_MAX), value_(0.0)
{
}

DoubleRange::~DoubleRange()
{
}

void DoubleRange::init(const std::string& s, bool* ok)
{
    min_   = -DBL_MAX;
    max_   =  DBL_MAX;
    value_ = 0.0;

    // Extract range from the parameter CSV string.
    // Parameters, p, for DoubleRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -DBL_MAX to DBL_MAX)
    //  - With 1 entry the range is (p[0] to DBL_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    //
    // Notes: if p[0] is the string 'MIN' or p[1] is the string 'MAX'
    // these will resolve as -DBL_MAX and DBL_MAX respectively.
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u) {
        if (ok) *ok = false;
        return;
    }
    else if (p.size() == 1u) {
        if (p[0] == "MIN") min_ = -DBL_MAX;
        else min_ = oskar_settings_utility_string_to_double(p[0], ok);
    }
    else if (p.size() == 2u) {
        if (p[0] == "MIN")
            min_ = -DBL_MAX;
        else
            min_ = oskar_settings_utility_string_to_double(p[0], ok);
        if (p[1] == "MAX")
            max_ = DBL_MAX;
        else
            max_ = oskar_settings_utility_string_to_double(p[1], ok);
    }
    else {
        if (ok) *ok = false;
    }
}

void DoubleRange::fromString(const std::string& s, bool* ok)
{
    using namespace std;
    if (ok) *ok = false;

    double d = oskar_settings_utility_string_to_double(s, ok);
    if (ok && !*ok) return;

    if (d >= min_ && d <= max_) {
        if (ok) *ok = true;
        value_ = d;
    }
    else if (d < min_) { if (ok) *ok = false; value_ = min_; }
    else if (d > max_) { if (ok) *ok = false; value_ = max_; }
}

std::string DoubleRange::toString() const
{
    return oskar_settings_utility_double_to_string(value_, -17);
}

std::string DoubleRange::toString(const std::string& fmt) const
{
    return oskar_format_string(fmt, value_);
}

} // namespace oskar

