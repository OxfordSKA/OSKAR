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

#include "settings/types/oskar_Double.h"
#include "settings/oskar_settings_utility_string.h"
#include <cfloat>
#include <cmath>

namespace oskar {

Double::Double()
{
    (void) init(0);
}

Double::~Double()
{
}

bool Double::init(const char* /*s*/)
{
    format_ = AUTO;
    default_ = 0.0;
    value_ = 0.0;
    str_default_ = "0.0";
    str_value_ = "0.0";
    return true;
}

bool Double::set_default(const char* value)
{
    bool ok = true;
    std::string val(value);
    double d = oskar_settings_utility_string_to_double(val, &ok);
    if (!ok) return false;
    format_ = (val.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    default_ = d;
    str_default_ = oskar_settings_utility_double_to_string_2(default_,
            (format_ == AUTO ? 'g' : 'e'));
    set_value(value);
    return true;
}

bool Double::set_value(const char* value)
{
    bool ok = true;
    std::string val(value);
    double d = oskar_settings_utility_string_to_double(val, &ok);
    if (!ok) return false;
    format_ = (val.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    value_ = d;
    str_value_ = oskar_settings_utility_double_to_string_2(value_,
            (format_ == AUTO ? 'g' : 'e'));
    return true;
}

bool Double::is_default() const
{
    return (fabs(value_ - default_) < DBL_EPSILON);
}

double Double::value() const
{
    return value_;
}

double Double::default_value() const
{
    return default_;
}

bool Double::operator==(const Double& other) const
{
    return (fabs(value_ - other.value_) < DBL_EPSILON);
}

bool Double::operator>(const Double& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
