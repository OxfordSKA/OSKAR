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

#include <oskar_Time.hpp>
#include <oskar_settings_utility_string.hpp>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cfloat>
#include <iostream>
using namespace std;
namespace oskar {

Time::Time()
{
}

Time::~Time()
{
}

bool Time::init(const std::string& /*s*/)
{
    default_.clear();
    value_.clear();
    return true;
}

bool Time::set_default(const std::string& value)
{
    bool ok = from_string_(value, default_);
    if (ok) {
        value_ = default_;
    }
    else {
        default_.clear();
        value_.clear();
    }
    return ok;
}

std::string Time::get_default() const
{
    return to_string_(default_);
}

bool Time::set_value(const std::string& value)
{
    return from_string_(value, value_);
}

std::string Time::get_value() const
{
    return to_string_(value_);
}

bool Time::is_default() const
{
    bool answer = true;
    answer &= (value_.hours == default_.hours);
    answer &= (value_.minutes == default_.minutes);
    answer &= (fabs(value_.seconds - default_.seconds) < DBL_MIN);
    return answer;
}

double Time::to_seconds() const
{
    if (value_.seconds > 60.) return value_.seconds;
    return (value_.hours * 3600.0) + (value_.minutes * 60.0) + value_.seconds;
}

bool Time::operator==(const Time& other) const
{
    return (fabs(to_seconds() - other.to_seconds()) < DBL_MIN);
}

bool Time::operator>(const Time& other) const
{
    return to_seconds() > other.to_seconds();
}

bool Time::from_string_(const std::string& s, Value& value) const
{
    bool ok = true;
    if (s.empty())
    {
        value.clear();
        return false;
    }
    // s.zzzzzzzzz
    else if (s.find(":") == std::string::npos)
    {
        double seconds = oskar_settings_utility_string_to_double(s, &ok);
        value.hours = floor(seconds / 3600.);
        value.minutes = floor((seconds - (value.hours * 3600.)) / 60.);
        value.seconds = seconds - (value.hours * 3600.) - (value.minutes * 60.);
        value.format = SECONDS;
    }
    // hh:mm:ss.zzzzzzzzz
    else
    {
        std::istringstream ss(s);
        std::string token;
        std::getline(ss, token, ':');
        value.hours = oskar_settings_utility_string_to_int(token);
        if (value.hours < 0) {
            value.hours = 0;
            return false;
        }
        std::getline(ss, token, ':');
        value.minutes = oskar_settings_utility_string_to_int(token);
        if (value.minutes < 0 || value.minutes >= 59) {
            value.minutes = 0;
            return false;
        }
        std::getline(ss, token);
        value.seconds = oskar_settings_utility_string_to_double(token, &ok);
        if (value.minutes < 0.0 || value.minutes >= 60.0) {
            value.seconds = 0.0;
            return false;
        }
        value.format = TIME_STRING;
    }
    return true;
}

std::string Time::to_string_(const Value& value) const
{
    std::ostringstream ss;
    if (value.format == UNDEF)
    {
        return std::string();
    }
    else if (value.format == TIME_STRING)
    {
        char sep = ':';
        ss << std::setfill('0') << std::setw(2) << value.hours << sep;
        ss << std::setfill('0') << std::setw(2) << value.minutes << sep;
        if (value.seconds < 10.)
            ss << 0;
        ss << oskar_settings_utility_double_to_string_2(value.seconds, 'f', 12);
    }
    else if (value.format == SECONDS)
    {
        double seconds = value.hours * 3600. + value.minutes * 60.
                        + value.seconds;
        ss << oskar_settings_utility_double_to_string_2(seconds, 'f', 12);
    }
    return ss.str();
}

} // namespace oskar

