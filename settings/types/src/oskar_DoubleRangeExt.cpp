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

namespace oskar {

DoubleRangeExt::DoubleRangeExt()
: min_(-DBL_MAX), max_(DBL_MAX), value_(0.0)
{
}

DoubleRangeExt::~DoubleRangeExt()
{
}

void DoubleRangeExt::init(const std::string& s, bool* ok)
{
    if (*ok) *ok = true;
    // Extract range from the parameter CSV string.
    // Parameters, p, for DoubleRangeExt should be length 3 or 4.
    //  - With 3 entries the range is (p[0] to p[1]) with an extended minimum
    //    value of p[2]
    //  - With 4 entries the range is (p[0] to p[1]) with an extended minimum
    //    value of p[2] and an extended maximum value of p[3]
    //  - For the double range parameters, p[0] and p[1], special values
    //    of 'MIN' and 'MAX' map to -DBL_MAX and DBL_MIN respectively.
    std::vector<std::string> p;
    p = oskar_settings_utility_string_get_type_params(s);
    //std::cout << p.size() << std::endl;
    if (p.size() < 3u || p.size() > 4u) {
        if (*ok) *ok = false;
        return;
    }
    if (p[0] == "MIN") min_ = -DBL_MAX;
    else min_ = oskar_settings_utility_string_to_double(p[0], ok);
    if (p[1] == "MAX") min_ =  DBL_MAX;
    else max_ = oskar_settings_utility_string_to_double(p[1], ok);
    ext_min_ = p[2];
    if (p.size() == 4u) {
        ext_max_ = p[3];
    }


    // Reset the value.
    value_ = 0.0;
}

void DoubleRangeExt::fromString(const std::string& s, bool* ok)
{
    if (ok) *ok = true;

    if (s.empty()) {
        if (ok) *ok = false;
        return;
    }

    if (s == ext_min_) {
        value_ = ext_min_;
    }
    else if (s == ext_max_) {
        value_ = ext_max_;
    }
    else {
        double v = oskar_settings_utility_string_to_double(s, ok);
        if (ok && !*ok) return;
        if (v < min_ and !ext_min_.empty()) {
            value_ = ext_min_;
        }
        else if (v > max_ and !ext_max_.empty()) {
            value_ = ext_max_;
        }
        else {
            if (v >= max_) value_ = max_;
            else if (v <= min_) value_ = min_;
            else value_ = v;
        }
    }
}

std::string DoubleRangeExt::toString() const
{
    using namespace ttl::var;

    if (value_.is_singular()) return std::string();

    if (value_.which() == DOUBLE) {
        double v = get<double>(value_);
        std::string s = oskar_settings_utility_double_to_string(v, -17);
        return s;
    }
    else if (value_.which() == STRING) {
        return get<std::string>(value_);
    }

    return std::string();
}

} // namespace oskar

