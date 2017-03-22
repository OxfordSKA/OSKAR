/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#include "settings/types/oskar_Time.h"
#include "settings/oskar_settings_utility_string.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <iostream>

using namespace std;

namespace oskar {

bool Time::init(const string& /*s*/)
{
    default_.clear();
    value_.clear();
    return true;
}

bool Time::set_default(const string& val)
{
    bool ok = from_string(val, default_);
    if (ok) {
        value_ = default_;
    }
    else {
        default_.clear();
        value_.clear();
    }
    return ok;
}

string Time::get_default() const
{
    return to_string(default_);
}

bool Time::set_value(const string& val)
{
    return from_string(val, value_);
}

string Time::get_value() const
{
    return to_string(value_);
}

bool Time::is_default() const
{
    bool answer = true;
    answer &= (value_.hours == default_.hours);
    answer &= (value_.minutes == default_.minutes);
    answer &= (fabs(value_.seconds - default_.seconds) < 1e-6);
    return answer;
}

double Time::to_seconds() const
{
    if (value_.seconds > 60.) return value_.seconds;
    return (value_.hours * 3600.0) + (value_.minutes * 60.0) + value_.seconds;
}

bool Time::operator==(const Time& other) const
{
    return (fabs(to_seconds() - other.to_seconds()) < 1e-6);
}

bool Time::operator>(const Time& other) const
{
    return to_seconds() > other.to_seconds();
}

bool Time::from_string(const string& s, Value& val)
{
    bool ok = true;
    if (s.empty())
    {
        val.clear();
        return false;
    }
    // s.zzzzzzzzz
    else if (s.find(":") == string::npos)
    {
        double seconds = oskar_settings_utility_string_to_double(s, &ok);
        val.hours = floor(seconds / 3600.);
        val.minutes = floor((seconds - (val.hours * 3600.)) / 60.);
        val.seconds = seconds - (val.hours * 3600.) - (val.minutes * 60.);
        val.format = SECONDS;
    }
    // hh:mm:ss.zzzzzzzzz
    else
    {
        istringstream ss(s);
        string token;
        getline(ss, token, ':');
        val.hours = oskar_settings_utility_string_to_int(token);
        if (val.hours < 0) {
            val.hours = 0;
            return false;
        }
        getline(ss, token, ':');
        val.minutes = oskar_settings_utility_string_to_int(token);
        if (val.minutes < 0 || val.minutes > 59) {
            val.minutes = 0;
            return false;
        }
        getline(ss, token);
        val.seconds = oskar_settings_utility_string_to_double(token, &ok);
        if (val.seconds < 0.0 || val.seconds >= 60.0) {
            val.seconds = 0.0;
            return false;
        }
        val.format = TIME_STRING;
    }
    return true;
}

string Time::to_string(const Value& val)
{
    ostringstream ss;
    if (val.format == UNDEF)
    {
        return string();
    }
    else if (val.format == TIME_STRING)
    {
        char sep = ':';
        ss << setfill('0') << setw(2) << val.hours << sep;
        ss << setfill('0') << setw(2) << val.minutes << sep;
        if (val.seconds < 10.)
            ss << 0;
        ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
    }
    else if (val.format == SECONDS)
    {
        double seconds = val.hours * 3600. + val.minutes * 60. + val.seconds;
        ss << oskar_settings_utility_double_to_string_2(seconds, 'f', 12);
    }
    return ss.str();
}

} // namespace oskar

