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

namespace oskar {

IntRangeExt::IntRangeExt()
: min_(-INT_MAX), max_(INT_MAX), default_(0), value_(0)
{
}

IntRangeExt::~IntRangeExt()
{
}

bool IntRangeExt::init(const std::string& s)
{
    ext_min_.clear();
    ext_max_.clear();
    min_ = -INT_MAX;
    max_ =  INT_MAX;
    value_ = 0;
    default_ = 0;

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
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
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


bool IntRangeExt::set_default(const std::string &value)
{
    bool ok = from_string_(default_, value);
    if (ok) {
        value_ = default_;
    }
    return ok;
}

std::string IntRangeExt::get_default() const
{
    return to_string_(default_);
}

bool IntRangeExt::set_value(const std::string& value)
{
    return from_string_(value_, value);
}

std::string IntRangeExt::get_value() const
{
    return to_string_(value_);
}

bool IntRangeExt::is_default() const
{
    if (value_.is_singular() || default_.is_singular()) return false;
    if (value_.which() == default_.which()) {
        if (value_.which() == INT) {
            return (ttl::var::get<int>(value_) == ttl::var::get<int>(default_));
        }
        else if (value_.which() == STRING) {
            return (ttl::var::get<std::string>(value_)
                            == ttl::var::get<std::string>(default_));
        }
    }
    return false;
}

bool IntRangeExt::set_value(int i)
{
    return from_int_(value_, i);
}

bool IntRangeExt::set_default(int i)
{
    return from_int_(default_, i);
}

bool IntRangeExt::operator==(const IntRangeExt& other) const
{
    using ttl::var::get;
    if (value_.is_singular() || other.value_.is_singular())
        return false;
    if (value_.which() == other.value_.which()) {
        if (value_.which() == INT) {
            return (get<int>(value_) == get<int>(default_));
        }
        else if (value_.which() == STRING) {
            return (get<std::string>(value_) == get<std::string>(default_));
        }
    }
    return false;
}

bool IntRangeExt::operator>(const IntRangeExt& other) const
{
    using ttl::var::get;
    if (value_.is_singular() || other.value_.is_singular())
        return false;
    if (value_.which() == other.value_.which()) {
        if (value_.which() == INT) {
            return (get<int>(value_) > get<int>(default_));
        }
        else if (value_.which() == STRING) {
            return false;
        }
    }
    return false;
}

bool IntRangeExt::from_string_(Value& value, const std::string& s) const
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
        else if (i < min_) {
            value = ext_min_;
            return false;
        }
        else if (i > max_ && !ext_max_.empty()) {
            value = ext_max_;
            return false;
        }
        else {
            return false;
        }
    }
}

bool IntRangeExt::from_int_(Value& value, int i) const
{
    if (i < min_ && !ext_min_.empty()) {
        value = ext_min_;
        return false;
    }
    else if (i > max_ && !ext_max_.empty()) {
        value = ext_max_;
        return false;
    }
    else {
        if (i >= max_) {
            value = max_;
            return false;
        }
        else if (i <= min_) {
            value = min_;
            return true;
        }
        else value = i;
    }
    return true;
}

std::string IntRangeExt::to_string_(const Value& value) const
{
    if (value.is_singular()) return std::string();
    if (value.which() == INT) {
        return oskar_settings_utility_int_to_string(ttl::var::get<int>(value));
    }
    else if (value.which() == STRING) {
        return ttl::var::get<std::string>(value);
    }
    return std::string();
}

} // namespace oskar

