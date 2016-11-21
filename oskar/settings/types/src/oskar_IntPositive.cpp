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
#include <oskar_IntPositive.hpp>

namespace oskar {

IntPositive::IntPositive() : default_(1), value_(1)
{
}

IntPositive::~IntPositive()
{
}

bool IntPositive::init(const std::string& /*s*/)
{
    default_ = 1;
    value_ = 1;
    return true;
}

bool IntPositive::set_default(const std::string& value)
{
    bool ok = from_string_(value, default_);
    if (ok) {
        value_ = default_;
    }
    return ok;
}

std::string IntPositive::get_default() const
{
    return oskar_settings_utility_int_to_string(default_);
}

bool IntPositive::set_value(const std::string& value)
{
    return from_string_(value, value_);
}

std::string IntPositive::get_value() const
{
    return oskar_settings_utility_int_to_string(value_);
}

bool IntPositive::is_default() const
{
    return default_ == value_;
}

bool IntPositive::from_string_(const std::string& s, int& value) const
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;
    if (i >= 1) {
        value = i;
        return true;
    }
    return false;
}

bool IntPositive::operator==(const IntPositive& other) const
{
    return value_ == other.value_;
}

bool IntPositive::operator>(const IntPositive& other) const
{
    return value_ > other.value_;
}

} // namespace oskar

