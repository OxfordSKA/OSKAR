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

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_IntRange.h"

#include <climits>
#include <vector>

using namespace std;

namespace oskar {

IntRange::IntRange()
{
    (void) init("");
}

IntRange::~IntRange()
{
}

bool IntRange::init(const char* s)
{
    min_ = -INT_MAX;
    max_ =  INT_MAX;
    default_ = 0;
    value_ = 0;
    str_default_ = "0";
    str_value_ = "0";

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -INT_MAX to INT_MAX)
    //  - With 1 entry the range is (p[0] to INT_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    //
    // Notes: if p[0] is the string 'MIN' or p[1] is the string 'MAX'
    // these will resolve as -INT_MAX and INT_MAX respectively.
    //
    bool ok = true;
    vector<string> p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u)
        return true;
    else if (p.size() == 1u) {
        if (p[0] == "MIN") min_ = -INT_MAX;
        else min_ = oskar_settings_utility_string_to_int(p[0], &ok);
        return true;
    }
    else if (p.size() == 2u) {
        if (p[0] == "MIN") min_ = -INT_MAX;
        else min_ = oskar_settings_utility_string_to_int(p[0], &ok);
        if (!ok) return false;
        if (p[1] == "MAX") max_ = INT_MAX;
        else max_ = oskar_settings_utility_string_to_int(p[1], &ok);
        return true;
    }

    // If more than 3 parameters, set the status to false.
    return false;
}

bool IntRange::set_default(const char* value)
{
    bool ok = from_string_(value, default_);
    str_default_ = oskar_settings_utility_int_to_string(default_);
    if (ok)
        set_value(value);
    return ok;
}

bool IntRange::set_value(const char* value)
{
    bool ok = from_string_(value, value_);
    str_value_ = oskar_settings_utility_int_to_string(value_);
    return ok;
}

bool IntRange::is_default() const
{
    return (value_ == default_);
}

int IntRange::value() const
{
    return value_;
}

int IntRange::default_value() const
{
    return default_;
}

int IntRange::min() const
{
    return min_;
}

int IntRange::max() const
{
    return max_;
}

bool IntRange::operator==(const IntRange& other) const
{
    return value_ == other.value_;
}

bool IntRange::operator>(const IntRange& other) const
{
    return value_ > other.value_;
}

bool IntRange::from_string_(const string& s, int& value) const
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;

    if (i >= min_ && i <= max_) {
        value = i;
        return true;
    }
    else if (i < min_) value = min_;
    else if (i > max_) value = max_;
    return false;
}

} // namespace oskar
