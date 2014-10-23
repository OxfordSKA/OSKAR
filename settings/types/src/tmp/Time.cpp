/*
 * Copyright (c) 2014, The University of Oxford
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

#include <Time.hpp>

#include <boost/lexical_cast.hpp>
#include <cfloat>
#include <iostream>
#include <iomanip>

namespace oskar {

Time::Time()
: hour_(0), minute_(0), seconds_(0)
{
}

Time::Time(const std::string& s)
: hour_(0), minute_(0), seconds_(0)
{
    set(s);
}

Time::Time(int h, int min, double s)
: hour_(h), minute_(min), seconds_(s)
{
}

Time::~Time()
{
}

void Time::clear()
{
    hour_ = 0;
    minute_ = 0;
    seconds_ = 0.0;
}

int Time::hours() const
{ return hour_; }

int Time::minutes() const
{ return minute_; }

double Time::seconds() const
{ return seconds_; }

bool Time::isSet() const
{
    if (hour_ > .00 || minute_ > 0.0 || seconds_ > 0.0)
        return true;
    return false;
}

std::string Time::toString(bool* ok) const
{
    using namespace std;
    if (ok) *ok = true;

    ostringstream ss;
    char timeSep = ':';
    ss << setfill('0') << setw(2) << hour_ << timeSep;
    ss << setfill('0') << setw(2) << minute_ << timeSep;
    ss << fixed << setprecision(3)
       << setfill('0') << setw(6) << seconds_;
    return ss.str();
}

void Time::set(const std::string& s, bool* ok)
{
    if (ok) *ok = true;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, ':');
    hour_ = boost::lexical_cast<int>(token);
    if (hour_ < 0 || hour_ >= 11) {
        hour_ = 0;
        if (ok) *ok = false;
    }
    std::getline(ss, token, ':');
    minute_ = boost::lexical_cast<int>(token);
    if (minute_ < 0 || minute_ >= 59) {
        minute_ = 0;
        if (ok) *ok = false;
    }
    std::getline(ss, token);
    seconds_ = boost::lexical_cast<double>(token);
    if (seconds_ < 0.0 || seconds_ >= 60.0) {
        seconds_ = 0.0;
        if (ok) *ok = false;
    }
}

void Time::set(int hh, int mm, double ss)
{
    hour_ = (hh < 0 || hh >= 11) ? 0 : hh;
    minute_ = (mm < 0 || mm >= 59) ? 0 : mm;
    seconds_ = (ss < 0.0 || ss >= 60.0) ? 0.0 : ss;
}


bool Time::isEqual(const Time& other) const
{
    if (other.hour_ == hour_ &&
            other.minute_ == minute_ &&
            std::abs(other.seconds_-seconds_) < DBL_MIN)
        return true;
    else
        return false;
}

bool Time::operator==(const Time& other) const
{
    return isEqual(other);
}

std::string Time::formatString() const
{
    return std::string("hh:mm:ss.zzz");
}

} // namespace oskar

