/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_Time.h"
#include "settings/oskar_settings_utility_string.h"
#include <sstream>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <cfloat>

using namespace std;

namespace oskar {

static bool from_string(const string& s, Time::Value& val)
{
    bool ok = true;
    if (s.empty())
    {
        //val.clear();
        return false;
    }
    // s.zzzzzzzzz
    else if (s.find(":") == string::npos)
    {
        double seconds = oskar_settings_utility_string_to_double(s, &ok);
        val.hours = floor(seconds / 3600.);
        val.minutes = floor((seconds - (val.hours * 3600.)) / 60.);
        val.seconds = seconds - (val.hours * 3600.) - (val.minutes * 60.);
        val.format = Time::SECONDS;
    }
    // hh:mm:ss.zzzzzzzzz
    else
    {
        istringstream ss(s);
        string token;
        getline(ss, token, ':');
        int hours = oskar_settings_utility_string_to_int(token);
        if (hours < 0) return false;
        getline(ss, token, ':');
        int minutes = oskar_settings_utility_string_to_int(token);
        if (minutes < 0 || minutes > 59) return false;
        getline(ss, token);
        double seconds = oskar_settings_utility_string_to_double(token, &ok);
        if (seconds < 0.0 || seconds >= 60.0) return false;
        val.hours = hours;
        val.minutes = minutes;
        val.seconds = seconds;
        val.format = Time::TIME_STRING;
    }
    return true;
}

static string to_string(const Time::Value& val)
{
    ostringstream ss;
    if (val.format == Time::UNDEF)
    {
        return string();
    }
    else if (val.format == Time::TIME_STRING)
    {
        char sep = ':';
        ss << setfill('0') << setw(2) << val.hours << sep;
        ss << setfill('0') << setw(2) << val.minutes << sep;
        if (val.seconds < 10.) ss << 0;
        ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
    }
    else if (val.format == Time::SECONDS)
    {
        double seconds = val.hours * 3600. + val.minutes * 60. + val.seconds;
        ss << oskar_settings_utility_double_to_string_2(seconds, 'f', 12);
    }
    return ss.str();
}

Time::Time()
{
    (void) Time::init(0);
}

// LCOV_EXCL_START
Time::~Time()
{
}
// LCOV_EXCL_STOP

bool Time::init(const char* /*s*/)
{
    default_.clear();
    value_.clear();
    str_default_ = to_string(default_);
    str_value_ = to_string(value_);
    return true;
} // LCOV_EXCL_LINE

bool Time::set_default(const char* val)
{
    bool ok = from_string(val, default_);
    str_default_ = to_string(default_);
    if (ok) set_value(val); else init(0);
    return ok;
} // LCOV_EXCL_LINE

bool Time::set_value(const char* val)
{
    bool ok = from_string(val, value_);
    str_value_ = to_string(value_);
    return ok;
} // LCOV_EXCL_LINE

bool Time::is_default() const
{
    bool answer = true;
    answer &= (value_.hours == default_.hours);
    answer &= (value_.minutes == default_.minutes);
    answer &= (fabs(value_.seconds - default_.seconds) < 1e-6);
    return answer;
}

Time::Value Time::value() const
{
    return value_;
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

} // namespace oskar
