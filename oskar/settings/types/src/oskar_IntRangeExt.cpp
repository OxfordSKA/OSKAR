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

#include "settings/types/oskar_IntRangeExt.h"
#include "settings/oskar_settings_utility_string.h"
#include <climits>
#include <vector>
#include <iostream>

using namespace std;
using ttl::var::get;

namespace oskar {

enum value_types { INT, STRING };

static string to_string(const IntRangeExt::Value& value)
{
    if (value.is_singular()) return string();
    if (value.which() == INT)
        return oskar_settings_utility_int_to_string(get<int>(value));
    else if (value.which() == STRING)
        return get<string>(value);
    return string();
}

static bool compare(const IntRangeExt::Value& a, const IntRangeExt::Value& b)
{
    if (a.is_singular() || b.is_singular()) return false;
    if (a.which() != b.which()) return false;
    if (a.which() == STRING) return (get<string>(a) == get<string>(b));
    if (a.which() == INT) return (get<int>(a) == get<int>(b));
    return false;
}

IntRangeExt::IntRangeExt()
{
    (void) init("");
}

IntRangeExt::~IntRangeExt()
{
}

bool IntRangeExt::init(const char* s)
{
    ext_min_.clear();
    ext_max_.clear();
    min_ = -INT_MAX;
    max_ =  INT_MAX;
    value_ = 0;
    default_ = 0;
    str_default_ = "0";
    str_value_ = "0";

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
    bool ok = true;
    vector<string> p = oskar_settings_utility_string_get_type_params(s);
    // For less than 3 values, just use IntRange instead!
    if (p.size() < 3u || p.size() > 4u) {
        return false;
    }
    else if (p[0] == "-INT_MAX" || p[0] == "-MAX" ||
                    p[0] == "INT_MIN" || p[0] == "MIN")
        min_ = -INT_MAX;
    else
        min_ = oskar_settings_utility_string_to_int(p[0], &ok);
    if (!ok) return false;
    if (p[1] == "INT_MAX" || p[1] == "MAX")
        max_ = INT_MAX;
    else
        max_ = oskar_settings_utility_string_to_int(p[1], &ok);
    ext_min_ = p[2];
    if (p.size() == 4u) {
        if (max_ == INT_MAX) return false;
        ext_max_ = p[3];
    }
    return ok;
}

bool IntRangeExt::set_default(const char* value)
{
    bool ok = from_string(default_, value);
    str_default_ = to_string(default_);
    if (ok)
        set_value(value);
    return ok;
}

bool IntRangeExt::set_value(const char* value)
{
    bool ok = from_string(value_, value);
    str_value_ = to_string(value_);
    return ok;
}

bool IntRangeExt::is_default() const
{
    return compare(value_, default_);
}

int IntRangeExt::value() const
{
    if (value_.which() == STRING && get<string>(value_) == ext_max_)
        return max_;
    if (value_.which() == STRING && get<string>(value_) == ext_min_)
        return min_;
    return get<int>(value_);
}

int IntRangeExt::min() const
{
    return min_;
}

int IntRangeExt::max() const
{
    return max_;
}

const char* IntRangeExt::ext_min() const
{
    return ext_min_.c_str();
}

const char* IntRangeExt::ext_max() const
{
    return ext_max_.c_str();
}

bool IntRangeExt::operator==(const IntRangeExt& other) const
{
    return compare(value_, other.value_);
}

bool IntRangeExt::operator>(const IntRangeExt& other) const
{
    if (value_.is_singular() || other.value_.is_singular()) return false;
    if (value_.which() == other.value_.which()) return false;
    if (value_.which() == STRING) return false;
    if (value_.which() == INT) return (get<int>(value_) > get<int>(default_));
    return false;
}

bool IntRangeExt::from_string(Value& value, const char* s) const
{
    // Catch cases where the range is being set with a special string.
    if (oskar_settings_utility_string_starts_with(ext_min_, s)) {
        value = ext_min_;
        return true;
    }
    else if (!ext_max_.empty() &&
                    oskar_settings_utility_string_starts_with(ext_max_, s)) {
        value = ext_max_;
        return true;
    }
    // Otherwise the string should contain an integer.
    else {
        bool ok = true;
        int i = oskar_settings_utility_string_to_int(s, &ok);
        if (i >= min_ && i <= max_) {
            value = i;
            return true;
        }
        // If the integer is out of range, set it to the closest special value.
        else if (i < min_) value = ext_min_;
        else if (i > max_ && !ext_max_.empty()) value = ext_max_;
    }
    return false;
}

} // namespace oskar
