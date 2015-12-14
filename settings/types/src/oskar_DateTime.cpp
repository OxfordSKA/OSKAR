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
#include <iostream>
#include <iomanip>
#include <oskar_DateTime.hpp>


namespace oskar {

DateTime::DateTime()
: style_(UNDEF), year_(0), month_(0), day_(0), hours_(0), minutes_(0), seconds_(0)
{
}

DateTime::~DateTime()
{
}

void DateTime::init(const std::string& /*s*/, bool* /*ok*/)
{
    style_ = UNDEF;
    year_ = 0;
    month_ = 0;
    day_ = 0;
    hours_ = 0;
    minutes_ = 0;
    seconds_ = 0.0;
}

void DateTime::parse_date_style_1_(const std::string& s, bool* ok)
{
//    std::cout << "FORMAT 1 = d-M-yyyy h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    day_ = oskar_settings_utility_string_to_int(token);
    if (day_ < 0 || day_ >= 31) {
        day_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token, '-');
    month_ = oskar_settings_utility_string_to_int(token);
    if (month_ < 0 || month_ >= 12) {
        month_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token, ' ');
    year_ = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token);
    parse_time_(token, ok);
}
void DateTime::parse_date_style_2_(const std::string& s, bool* ok)
{
    //std::cout << "FORMAT 2 = yyyy/M/d/h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '/');
    year_ = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '/');
    month_ = oskar_settings_utility_string_to_int(token);
    if (month_ < 0 || month_ >= 12) {
        month_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token, '/');
    day_ = oskar_settings_utility_string_to_int(token);
    if (day_ < 0 || day_ >= 31) {
        day_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token);
    parse_time_(token, ok);
}

void DateTime::parse_date_style_3_(const std::string& s, bool* ok)
{
//    std::cout << "FORMAT 3 = yyyy-M-d h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    year_ = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '-');
    month_ = oskar_settings_utility_string_to_int(token);
    if (month_ < 0 || month_ >= 12) {
        month_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token, ' ');
    day_ = oskar_settings_utility_string_to_int(token);
    if (day_ < 0 || day_ >= 31) {
        day_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token);
    parse_time_(token, ok);
}

void DateTime::parse_date_style_4_(const std::string& s, bool* ok)
{
//    std::cout << "FORMAT 4 = yyyy-M-dTh:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    year_ = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '-');
    month_ = oskar_settings_utility_string_to_int(token);
    if (month_ < 0 || month_ >= 12) {
        month_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token, 'T');
    day_ = oskar_settings_utility_string_to_int(token);
    if (day_ < 0 || day_ >= 31) {
        day_ = 0;
        if (ok) { *ok = false; return; }
    }
    std::getline(ss, token);
    parse_time_(token, ok);
}

void DateTime::parse_time_(const std::string& s, bool* ok)
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

void DateTime::fromString(const std::string& s, bool* ok)
{
    init("", ok);
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    size_t p = s.find("-");
    if (p != std::string::npos)
    {
        if (p <= 2) {
            style_ = BRITISH;
            parse_date_style_1_(s, ok);
        }
        else if (s.find("T") != std::string::npos) {
            style_ = ISO;
            parse_date_style_4_(s, ok);
        }
        else {
            style_ = INTERNATIONAL;
            parse_date_style_3_(s, ok);
        }
    }
    else if (s.find("/") != std::string::npos) {
        style_ = CASA;
        parse_date_style_2_(s, ok);
    }
    else {
        if (ok) *ok = false;
        return;
    }
}

std::string DateTime::toString() const
{
    std::ostringstream ss;
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    switch (style_)
    {
    case BRITISH:
    {
        ss << std::setfill('0') << std::setw(2) << day() << '-';
        ss << std::setfill('0') << std::setw(2) << month() << '-';
        ss << std::setfill('0') << std::setw(4) << year() << ' ';
        ss << std::setfill('0') << std::setw(2) << hours() << ':';
        ss << std::setfill('0') << std::setw(2) << minutes() << ':';
        if (seconds() < 10.) ss << 0;
        ss << oskar_settings_utility_double_to_string(seconds(), -10);
        break;
    }
    case CASA:
    {
        ss << std::setfill('0') << std::setw(4) << year() << '/';
        ss << std::setfill('0') << std::setw(2) << month() << '/';
        ss << std::setfill('0') << std::setw(2) << day() << '/';
        ss << std::setfill('0') << std::setw(2) << hours() << ':';
        ss << std::setfill('0') << std::setw(2) << minutes() << ':';
        if (seconds() < 10.) ss << 0;
        ss << oskar_settings_utility_double_to_string(seconds(), -10);
        break;
    }
    case INTERNATIONAL:
    {
        ss << std::setfill('0') << std::setw(4) << year() << '-';
        ss << std::setfill('0') << std::setw(2) << month() << '-';
        ss << std::setfill('0') << std::setw(2) << day() << ' ';
        ss << std::setfill('0') << std::setw(2) << hours() << ':';
        ss << std::setfill('0') << std::setw(2) << minutes() << ':';
        if (seconds() < 10.) ss << 0;
        ss << oskar_settings_utility_double_to_string(seconds(), -10);
        break;
    }
    case ISO:
    {
        ss << std::setfill('0') << std::setw(4) << year() << '-';
        ss << std::setfill('0') << std::setw(2) << month() << '-';
        ss << std::setfill('0') << std::setw(2) << day() << 'T';
        ss << std::setfill('0') << std::setw(2) << hours() << ':';
        ss << std::setfill('0') << std::setw(2) << minutes() << ':';
        if (seconds() < 10.) ss << 0;
        ss << oskar_settings_utility_double_to_string(seconds(), -10);
        break;
    }
    default:
        return std::string();
        break;
    }
    return ss.str();
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
    return hours_;
}

int DateTime::minutes() const
{
    return minutes_;
}

double DateTime::seconds() const
{
    return seconds_;
}

} // namespace oskar

