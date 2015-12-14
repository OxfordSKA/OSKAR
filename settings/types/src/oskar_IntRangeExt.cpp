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
#include <oskar_IntRangeExt.hpp>

namespace oskar {

IntRangeExt::IntRangeExt() : min_(-INT_MAX), max_(INT_MAX), value_(0)
{
}

IntRangeExt::~IntRangeExt()
{
}

void IntRangeExt::init(const std::string& s, bool* ok)
{
    if (*ok) *ok = true;

    smin_.clear();
    smax_.clear();
    min_ = -INT_MAX;
    max_ =  INT_MAX;

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRangeExt should be of length between 3 and 4
    //  - With 3 entries the range from (p[0] to p[1] with special string [2]
    //    for values < p[0]
    //  - With 4 entries the range from (p[0] to p[1] with special string p[2]
    //    for values < p[0] and special string p[3] for values > p[1]
    //
    // Note:
    // - p[0] can never be -INT_MAX and for 4 entries p[1] can't be bigger than
    //   INT_MAX-1
    // - For 3 entries, if p[1] is the string 'MAX' the max range is set to
    //   INT_MAX (note this is not possible when there is also a max string)
    //
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() < 3u) {
        if (*ok) *ok = false;
        return;
    }
    else if (p.size() == 3u) {
        min_ = oskar_settings_utility_string_to_int(p[0], ok);
        if (ok && !*ok) return;
        if (min_ == -INT_MAX && ok) { *ok = false; return; }
        if (p[1] == "MAX") max_ = INT_MAX;
        else max_ = oskar_settings_utility_string_to_int(p[1], ok);
        if (ok && !*ok) return;
        smin_ = p[2];
        smax_.clear();
        return;
    }
    else if (p.size() == 4u) {
        min_ = oskar_settings_utility_string_to_int(p[0], ok);
        if (ok && !*ok) return;
        if (min_ == -INT_MAX && ok) { *ok = false; return; }
        if (p[1] == "MAX" && ok) { *ok = false; return; }
        else max_ = oskar_settings_utility_string_to_int(p[1], ok);
        if (ok && !*ok) return;
        if (max_ == INT_MAX && ok) { *ok = false; return; }
        smin_ = p[2];
        smax_ = p[3];
        return;
    }

    // If more than 3 parameters, set the status to false.
    if (*ok) *ok = false;
}

void IntRangeExt::fromString(const std::string& s, bool* ok)
{
    if (ok) *ok = false;

    // Catch cases where the range is being set with a special string.
    if (oskar_settings_utility_string_starts_with(smin_, s)) {
        if (ok) *ok = true;
        value_ = min_-1;
    }
    else if (!smax_.empty() && oskar_settings_utility_string_starts_with(smax_, s)) {
        if (ok) *ok = true;
        value_ = max_+1;
    }
    // Otherwise the string should contain an integer.
    else {
        int i = oskar_settings_utility_string_to_int(s, ok);
        if (i >= min_ && i <= max_) {
            if (ok) *ok = true;
            value_ = i;
        }
        // If the integer is out of range, set it to the closest special value.
        else if (i < min_) {
            if (ok) *ok = false;
            value_ = min_-1;
        }
        else if (i > max_ && !smax_.empty()) {
            if (ok) *ok = false;
            value_ = max_+1;
        }
        else {
            if (ok) *ok = false;
            value_ = 0;
        }
    }
}

std::string IntRangeExt::toString() const
{
    if (value_ == min_-1) return smin_;
    else if (!smax_.empty() && value_ == max_+1) return smax_;
    return oskar_settings_utility_int_to_string(value_);
}

} // namespace oskar

