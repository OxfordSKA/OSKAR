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

#include <IntRange.hpp>

#include <oskar_settings_utility_string.hpp>

#include <climits>
#include <vector>

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

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -INT_MAX to INT_MAX)
    //  - With 1 entry the range is (p[0] to INT_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u) {
        return;
    }
    else if (p.size() == 1u) {
        min_ = oskar_settings_utility_string_to_int(p[0], ok);
        return;
    }
    else if (p.size() == 2u) {
        min_ = oskar_settings_utility_string_to_int(p[0], ok);
        if (ok && !*ok) return;
        max_ = oskar_settings_utility_string_to_int(p[1], ok);
        return;
    }

    // If more than 3 parameters, set the status to false.
    if (*ok) *ok = false;
}

void IntRange::set(const std::string& s, bool* ok)
{
    if (ok) *ok = false;
    int i = oskar_settings_utility_string_to_int(s, ok);
    if (i >= min_ && i <= max_) {
        if (ok) *ok = true;
        value_ = i;
    }
    else if (i < min_) value_ = min_;
    else if (i > max_) value_ = max_;
    //else value_ = 0;
}

std::string IntRange::toString(bool* ok) const
{
    if (ok) *ok = true;
    return oskar_settings_utility_int_to_string(value_);
}



#if 0
IntRange::IntRange()
: min_(-LONG_MAX), max_(LONG_MAX)
{
    assert(value_.is_singular());
}

IntRange::IntRange(long int min, long int max)
: min_(min), max_(max)
{
    assert(value_.is_singular());
}

IntRange::IntRange(int min, int max)
: min_(min), max_(max)
{
    assert(value_.is_singular());
}

IntRange::IntRange(int min, int max, int value)
: min_(min), max_(max)
{
    set(value);
}

IntRange::IntRange(int min, int max, const std::string& ext_min)
: min_(min), max_(max),  ext_min_(ext_min)
{
    assert(value_.is_singular());
}

IntRange::IntRange(int min, int max, const std::string& ext_min,
        const std::string& ext_max)
: min_(min), max_(max), ext_min_(ext_min), ext_max_(ext_max)
{
    assert(value_.is_singular());
}

bool IntRange::isSet() const
{
    return !value_.is_singular();
}

void IntRange::set(int i, bool* ok)
{
    if (ok) *ok = true;

    // If in range set to the specified value, otherwise set to
    // the value at the closest end of the range.
    if (i >= min_ && i <= max_) {
        value_ = static_cast<long int>(i);
    }
    // Note: cast to (long int) to avoids over/underflow issues.
    else if (std::abs(i-min_) < std::abs(max_-i)) {
        if (ext_min_.empty()) {
            if (ok) *ok = false;
            value_ = static_cast<long int>(min_);
        }
        else  {
            value_ = static_cast<std::string>(ext_min_);
        }
    }
    else {
        if (ext_max_.empty()) {
            if (ok) *ok = false;
            value_ = static_cast<long int>(max_);
        }
        else  {
            value_ = static_cast<std::string>(ext_max_);
        }
    }
}

void IntRange::set(const std::string& s, bool *ok)
{
    if (ok) *ok = true;
    if (s == ext_min_)
        value_ = static_cast<std::string>(ext_min_);
    else if (s == ext_max_)
        value_ = static_cast<std::string>(ext_max_);
    else {
        try {
            value_ = boost::lexical_cast<long int>(s);
        }
        catch (boost::bad_lexical_cast&) {
            if (ok) *ok = false;
            value_ = 0L;
        }
    }
}

int IntRange::getInt(bool* ok) const
{
    if (ok) *ok = true;

    if (value_.is_singular()) return 0;

    if (value_.which() == 0) { // Int
        return static_cast<int>(VAR::get<long int>(value_));
    }
    else if (value_.which() == 1) { // std::string
        std::string s = VAR::get<std::string>(value_);
        if (s == ext_min_) {
            if (ok) *ok = false;
            return (min_ == -INT_MAX) ? min_ : min_-1;
        }
        else if (s == ext_max_) {
            if (ok) *ok = false;
            return (max_ == INT_MAX) ? max_ : max_+1;
        }
        else {
            try {
                return boost::lexical_cast<int>(VAR::get<std::string>(value_));
            }
            catch (boost::bad_lexical_cast&) {
                if (ok) *ok = false;
                return 0;
            }
        }
    }
    else {
        if (ok) *ok = false;
        return 0;
    }
}

std::string IntRange::toString(bool* ok) const
{
    if (ok) *ok = true;

    if (value_.is_singular()) return std::string();

    if (value_.which() == 0) { // Int
        long int i = VAR::get<long int>(value_);
        return boost::lexical_cast<std::string>(i);
    }
    else if (value_.which() == 1) { // std::string
        return VAR::get<std::string>(value_);
    }
    else {
        if (ok) * ok = false;
        return std::string();
    }
}
#endif

} // namespace oskar

