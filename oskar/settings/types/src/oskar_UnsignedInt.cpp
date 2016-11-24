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

#include "oskar_settings_utility_string.hpp"
#include <sstream>
#include "oskar_UnsignedInt.hpp"

namespace oskar {

UnsignedInt::UnsignedInt()
: default_(0), value_(0)
{
}

UnsignedInt::~UnsignedInt()
{
}

bool UnsignedInt::init(const std::string& /*s*/)
{
    default_ = 0;
    value_ = 0;
    return true;
}

bool UnsignedInt::set_default(const std::string& value)
{
    bool ok = true;
    int i =  oskar_settings_utility_string_to_int(value, &ok);
    if (!ok || i < 0) {
        i = 0;
        return false;
    }
    default_ = i;
    value_ = default_;
    return true;
}

std::string UnsignedInt::get_default() const
{
    return oskar_settings_utility_int_to_string(default_);
}

bool UnsignedInt::set_value(const std::string& value)
{
    bool ok = true;
    int i =  oskar_settings_utility_string_to_int(value, &ok);
    if (!ok || i < 0) {
        i = 0;
        return false;
    }
    value_ = i;
    return true;
}

std::string UnsignedInt::get_value() const
{
    return oskar_settings_utility_int_to_string(value_);
}

bool UnsignedInt::is_default() const
{
    return value_ == default_;
}

bool UnsignedInt::operator==(const UnsignedInt& other) const
{
    return value_ == other.value_;
}

bool UnsignedInt::operator>(const UnsignedInt& other) const
{
    return value_ > other.value_;
}

} // namespace oskar

