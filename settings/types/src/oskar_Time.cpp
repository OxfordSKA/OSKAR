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
#include <sstream>
#include <iomanip>
#include <iostream>
#include <oskar_Time.hpp>


namespace oskar {

Time::Time()
: hours_(0), minutes_(0), seconds_(0)
{
}

Time::~Time()
{
}

void Time::init(const std::string& /*s*/, bool* /*ok*/)
{
    hours_ = 0;
    minutes_ = 0;
    seconds_ = 0.0;
}

void Time::fromString(const std::string& s, bool* ok)
{
    // s.zzzzzzzzz
    if (s.find(":") == std::string::npos)
    {
        double seconds = oskar_settings_utility_string_to_double(s, ok);
        seconds_ = seconds;
        hours_ = 0;
        minutes_ = 0;
    }
    // hh:mm:ss.zzzzzzzzz
    else
    {
        std::istringstream ss(s);
        std::string token;
        std::getline(ss, token, ':');
        hours_ = oskar_settings_utility_string_to_int(token);
        if (hours_ < 0 || hours_ >= 11) {
            hours_ = 0;
            if (ok) { *ok = false; return; }
        }
        std::getline(ss, token, ':');
        minutes_ = oskar_settings_utility_string_to_int(token);
        if (minutes_ < 0 || minutes_ >= 59) {
            minutes_ = 0;
            if (ok) { *ok = false; return; }
        }
        std::getline(ss, token);
        seconds_ = oskar_settings_utility_string_to_double(token, ok);
        if (minutes_ < 0.0 || minutes_ >= 60.0) {
            seconds_ = 0.0;
            if (ok) { *ok = false; return; }
        }
    }
}

std::string Time::toString() const
{
    std::ostringstream ss;
    char sep = ':';
    ss << std::setfill('0') << std::setw(2) << hours() << sep;
    ss << std::setfill('0') << std::setw(2) << minutes() << sep;
    if (seconds() < 10.)
        ss << 0;
    ss << oskar_settings_utility_double_to_string(seconds(), -10);
    return ss.str();
}

int Time::hours() const
{
    if (seconds_ > 3600.0) return int(seconds_/3600.);
    return hours_;
}

int Time::minutes() const
{

    if (seconds_ > 60.0) {
        return int((seconds_-hours()*3600)/60.0);
    }
    return minutes_;
}

double Time::seconds() const
{
    if (seconds_ > 60.0) {
        return seconds_ - hours()*3600 - minutes()*60;
    }
    return seconds_;
}

double Time::in_seconds() const
{
    if (seconds_ > 60.) return seconds_;
    return (hours_*3600.0) + (minutes_*60.0) + seconds_;
}


} // namespace oskar

