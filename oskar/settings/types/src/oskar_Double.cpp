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

#include "oskar_Double.hpp"
#include "oskar_settings_utility_string.hpp"
#include <sstream>
#include <cmath>
#include <cfloat>

namespace oskar {

Double::Double()
: format_(AUTO), default_(0.0), value_(0.0)
{
}

Double::~Double()
{
}

bool Double::init(const std::string& /*s*/)
{
    format_ = AUTO;
    default_ = 0.0;
    value_ = 0.0;
    return true;
}

bool Double::set_default(const std::string& value)
{
    bool ok = true;
    format_ = (value.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    default_ = oskar_settings_utility_string_to_double(value, &ok);
    if (ok) {
        value_ = default_;
    }
    else {
        value_ = 0.0;
        default_ = 0.0;
    }
    return ok;
}

std::string Double::get_default() const
{
    return oskar_settings_utility_double_to_string_2(default_,
                                             (format_ == AUTO ? 'g' : 'e'));
}

bool Double::set_value(const std::string& value)
{
    bool ok = true;
    format_ = (value.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    value_ = oskar_settings_utility_string_to_double(value, &ok);
    return ok;
}

std::string Double::get_value() const
{
    return oskar_settings_utility_double_to_string_2(value_,
                                             (format_ == AUTO ? 'g' : 'e'));
}

bool Double::is_default() const
{
    return (fabs(value_ - default_) < DBL_MIN);
}

bool Double::operator==(const Double& other) const
{
    return (fabs(value_ - other.value_) < DBL_MIN);
}

bool Double::operator>(const Double& other) const
{
    return value_ > other.value_;
}

} // namespace oskar

