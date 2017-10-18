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
#include "settings/types/oskar_RandomSeed.h"

namespace oskar {

static bool from_string(const char* s, int& value)
{
    if (oskar_settings_utility_string_starts_with("TIME", s, false)) {
        value = -1;
        return true;
    }
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;
    if (i < 1) return false;
    value = i;
    return true;
}

static std::string to_string(int value)
{
    if (value < 1) return "time";
    else return oskar_settings_utility_int_to_string(value);
}

RandomSeed::RandomSeed()
{
    (void) init(0);
}

RandomSeed::~RandomSeed()
{
}

bool RandomSeed::init(const char* /*s*/)
{
    default_ = 1;
    value_ = 1;
    str_default_ = "1";
    str_value_ = "1";
    return true;
}

bool RandomSeed::set_default(const char* s)
{
    bool ok = from_string(s, default_);
    str_default_ = to_string(default_);
    if (ok)
        set_value(s);
    else
        init(0);
    return ok;
}

bool RandomSeed::set_value(const char* s)
{
    bool ok = from_string(s, value_);
    str_value_ = to_string(value_);
    return ok;
}

bool RandomSeed::is_default() const { return value_ == default_; }

int RandomSeed::value() const
{
    return value_;
}

int RandomSeed::default_value() const
{
    return default_;
}

bool RandomSeed::operator==(const RandomSeed& other) const
{
    return value_ == other.value_;
}

bool RandomSeed::operator>(const RandomSeed& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
