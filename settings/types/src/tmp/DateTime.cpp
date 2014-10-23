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

#include <DateTime.hpp>

#include <boost/lexical_cast.hpp>
#include <cfloat>
#include <iostream>
#include <iomanip>

namespace oskar {

DateTime::DateTime()
: year_(0), month_(0), day_(0), hour_(0), minute_(0), seconds_(0)
{
}

DateTime::DateTime(const std::string& s)
: year_(0), month_(0), day_(0), hour_(0), minute_(0), seconds_(0)
{
    this->set(s);
}


DateTime::~DateTime()
{
}

void DateTime::clear()
{
    year_    = 0;
    month_   = 0;
    day_     = 0;
    hour_    = 0;
    minute_  = 0;
    seconds_ = 0.0;
}

int DateTime::year() const
{
    return year_;
}

int DateTime::month() const
{
    return month_;
}

int DateTime::day() const
{
    return day_;
}

int DateTime::hours() const
{
    return hour_;
}

int DateTime::minutes() const
{
    return minute_;
}

double DateTime::seconds() const
{
    return seconds_;
}

bool DateTime::isSet() const
{
    if (year_ > 0 && month_ > 0 && day_ > 0 &&
            hour_ > 0 && minute_ > 0 && seconds_ > 0.0)
        return true;
    else
        return false;
}

std::string DateTime::toString(bool* ok) const
{
    if (ok) *ok = true; // TODO return false if not set?
    std::ostringstream ss;
    char dateSep = '-';
    char timeSep = ':';
    ss << std::setfill('0') << std::setw(4) << year_ << dateSep;
    ss << std::setfill('0') << std::setw(2) << month_ << dateSep;
    ss << std::setfill('0') << std::setw(2) << day_ << " ";
    ss << std::setfill('0') << std::setw(2) << hour_ << timeSep;
    ss << std::setfill('0') << std::setw(2) << minute_ << timeSep;
    ss << std::fixed << std::setprecision(3) << std::setfill('0');
    ss << std::setw(6) << seconds_;

    std::string s = ss.str();
    return s;
}

void DateTime::set(int yr, int mo, int dd, int hh, int mm, double ss)
{
    year_    = yr;
    month_   = (mo < 1 || mo >= 12) ? 1 : mo;
    day_     = (dd < 1 || dd >= 31) ? 1 : dd;
    hour_    = (hh < 0 || hh >= 11) ? 0 : hh;
    minute_  = (mm < 0 || mm >= 59) ? 0 : mm;
    seconds_ = (ss < 0.0 || ss >= 60.0) ? 0.0 : ss;
}



void DateTime::set(const std::string& s, bool* ok)
{
    if (ok) *ok = true;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    year_ = boost::lexical_cast<int>(token);
    std::getline(ss, token, '-');
    month_ = boost::lexical_cast<int>(token);
    if (month_ < 1 || month_ >= 12) {
        month_ = 1;
        if (ok) *ok = false;
    }
    std::getline(ss, token, ' ');
    day_ = boost::lexical_cast<int>(token);
    if (day_ < 1 || day_ >= 31) {
        day_ = 1;
        if (ok) *ok = false;
    }
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

bool DateTime::isEqual(const DateTime& other) const
{
    if (other.year_ == year_ &&
            other.month_ == month_ &&
            other.day_ == day_ &&
            other.hour_ == hour_ &&
            other.minute_ == minute_ &&
            std::abs(other.seconds_-seconds_) < DBL_MIN)
        return true;
    else
        return false;
}

bool DateTime::operator==(const DateTime& other) const
{
    return isEqual(other);
}

std::string DateTime::formatString() const
{
    return std::string("yyyy-MM-dd hh:mm:ss.zzz");
}


} // namespace oskar

