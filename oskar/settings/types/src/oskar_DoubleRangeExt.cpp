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
#include <oskar_DoubleRangeExt.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <iostream>

using namespace std;

namespace oskar {

DoubleRangeExt::DoubleRangeExt()
: min_(-DBL_MAX), max_(DBL_MAX), format_(AUTO), value_(0.0), default_(0.0)
{
}

DoubleRangeExt::~DoubleRangeExt()
{
}

bool DoubleRangeExt::init(const std::string& s)
{
    // Reset the value.
    ext_min_.clear();
    ext_max_.clear();
    value_ = 0.0;
    default_ = 0.0;

    // Extract range from the parameter CSV string.
    // Parameters, p, for DoubleRangeExt should be length 3 or 4.
    //  - With 3 entries the range is (p[0] to p[1]) with an extended minimum
    //    value of p[2]
    //  - With 4 entries the range is (p[0] to p[1]) with an extended minimum
    //    value of p[2] and an extended maximum value of p[3]
    //  - For the double range parameters, p[0] and p[1], special values
    //    of 'MIN' and 'MAX' map to -DBL_MAX and DBL_MIN respectively.
    bool ok = true;
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() < 3u || p.size() > 4u) {
        return false;
    }
    if (p[0] == "-DBL_MAX" || p[0] == "-MAX")
        min_ = -DBL_MAX;
    else if (p[0] == "DBL_MIN" || p[0] == "-DBL_MIN"
                    || p[0] == "MIN" || p[0] == "-MIN")
        min_ = -DBL_MIN;
    else
        min_ = oskar_settings_utility_string_to_double(p[0], &ok);
    if (!ok) return false;
    if (p[1] == "DBL_MAX" || p[1] == "MAX")
        max_ =  DBL_MAX;
    else
        max_ = oskar_settings_utility_string_to_double(p[1], &ok);
    ext_min_ = p[2];
    if (p.size() == 4u) {
        ext_max_ = p[3];
    }
    return ok;
}

bool DoubleRangeExt::set_default(const std::string& value)
{
    bool ok = from_string_(default_, value);
    if (ok) {
        if (default_.which() == DOUBLE)
            format_ = (value.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
        value_ = default_;
    }
    return ok;
}

std::string DoubleRangeExt::get_default() const
{
    return to_string_(default_);
}

bool DoubleRangeExt::set_value(const std::string& value)
{
    bool ok = from_string_(value_, value);
    if (value_.which() == DOUBLE)
        format_ = (value.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    return ok;
}

std::string DoubleRangeExt::get_value() const
{
    return to_string_(value_);
}

bool DoubleRangeExt::is_default() const
{
    if (value_.is_singular() || default_.is_singular()) return false;
    if (value_.which() == default_.which()) {
        if (value_.which() == DOUBLE) {
            if (fabs(ttl::var::get<double>(value_) -
                     ttl::var::get<double>(default_)) < DBL_MIN) {
                return true;
            }
        }
        else if (value_.which() == STRING) {
            return ttl::var::get<std::string>(value_) ==
                            ttl::var::get<std::string>(default_);
        }
    }
    return false;
}

bool DoubleRangeExt::set_value(double d)
{
    return from_double_(value_, d);
}

bool DoubleRangeExt::set_default(double d)
{
    bool ok = from_double_(default_, d);
    if (ok) {
        value_ = default_;
    }
    return ok;
}

bool DoubleRangeExt::operator==(const DoubleRangeExt& other) const
{
    if (value_.is_singular() || other.value_.is_singular())
        return false;

    if (value_.which() == other.value_.which()) {
        if (value_.which() == DOUBLE) {
            return (fabs(ttl::var::get<double>(value_) -
                         ttl::var::get<double>(other.value_)) < DBL_MIN);
        }
        else if (value_.which() == STRING) {
            return ttl::var::get<std::string>(value_) ==
                            ttl::var::get<std::string>(other.value_);
        }
    }
    return false;
}

bool DoubleRangeExt::operator>(const DoubleRangeExt& other) const
{
    if (value_.is_singular() || other.value_.is_singular())
        return false;
    if (value_.which() == other.value_.which()) {
        if (value_.which() == DOUBLE) {
            return ttl::var::get<double>(value_) >
                ttl::var::get<double>(other.value_);
        }
        else if (value_.which() == STRING) {
            return false;
        }
    }
    return false;
}

bool DoubleRangeExt::from_double_(Value& value, double d) const
{
    if (d < min_ && !ext_min_.empty()) {
        value = ext_min_;
    }
    else if (d > max_ && !ext_max_.empty()) {
        value = ext_max_;
    }
    else {
        if (d >= max_) value = max_;
        else if (d <= min_) value = min_;
        else value = d;
    }
    return true;
}

bool DoubleRangeExt::from_string_(Value& value, const std::string& s) const
{
    if (s.empty())
        return false;

    if (s == ext_min_) {
        value = ext_min_;
    }
    else if (s == ext_max_) {
        value = ext_max_;
    }
    else {
        bool ok = true;
        double v = oskar_settings_utility_string_to_double(s, &ok);
        if (!ok) return false;
        if (v < min_ && !ext_min_.empty()) {
            value = ext_min_;
        }
        else if (v > max_ && !ext_max_.empty()) {
            value = ext_max_;
        }
        else {
            if (v >= max_) value = max_;
            else if (v <= min_) value = min_;
            else value = v;
        }
    }
    return true;
}

std::string DoubleRangeExt::to_string_(const Value& value) const
{
    if (value.is_singular()) return std::string();
    if (value.which() == DOUBLE) {
        double v = ttl::var::get<double>(value);
        int n = 16;
        if (v != 0.0 && v > 1.0) {
            n -= (floor(log10(v)) + 1);
        }
        std::string s = oskar_settings_utility_double_to_string_2(v, format_ == AUTO ? 'g' : 'e');
        return s;
    }
    else if (value.which() == STRING) {
        return ttl::var::get<std::string>(value);
    }
    return std::string();
}

} // namespace oskar

