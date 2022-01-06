/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_DateTime.h"
#include "settings/oskar_settings_utility_string.h"
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace std;

namespace oskar {

/**
 * @brief
 * Convert Modified Julian Day to date.
 *
 * @details
 *  Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
 *       4th ed., Duffet-Smith and Zwart, 2011.
 */
static void value_from_mjd(double mjd, DateTime::Value& val)
{
    double iMJD = 0.0;
    double F = modf(mjd, &iMJD);
    int I = static_cast<int>(iMJD) + 2400001;
    int A = static_cast<int>(trunc((I - 1867216.25) / 36524.25));
    int B = 0;
    if (I > 2299160)
    {
        B = I + 1 + A - static_cast<int>(trunc(A / 4.));
    }
    else
    {
        B = I;
    }

    int C = B + 1524;
    int D = static_cast<int>(trunc((C - 122.1) / 365.25));
    int E = static_cast<int>(trunc(365.25 * D));
    int G = static_cast<int>(trunc((C - E) / 30.6001));
    double day = C - E + F - static_cast<int>(trunc(30.6001 * G));

    int month = 0, year = 0;
    if (G < 13.5) month = G - 1; else month = G - 13;
    if (month > 2.5) year = D - 4716; else year = D - 4715;

    double iday = 0.0, fday = 0.0;
    fday = modf(day, &iday);
    double hours = trunc(fday * 24.);
    double minutes = trunc(((fday * 24.) - hours) * 60.);
    double seconds = ((fday * 24.0) - hours - (minutes / 60.)) * 3600.0;
    val.year = static_cast<int>(year);
    val.month = static_cast<int>(month);
    val.day = iday;
    val.hours = static_cast<int>(hours);
    val.minutes = static_cast<int>(minutes);
    val.seconds = seconds;
    val.style = DateTime::MJD;
}

static double value_to_mjd(const DateTime::Value& val)
{
    // Compute Julian Day Number (Note: all integer division).
    int a = (14 - val.month) / 12;
    int y = val.year + 4800 - a;
    int m = val.month + 12 * a - 3;
    int jdn = val.day + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
                            + (y / 400) - 32045;

    // Compute day fraction.
    double hours = val.hours + val.minutes / 60.0 + val.seconds / 3600.0;
    double day_fraction = hours / 24.0;
    day_fraction -= 0.5;
    return (jdn - 2400000.5) + day_fraction;
}

static bool parse_time(const string& s, DateTime::Value& val)
{
    istringstream ss(s);
    string token;
    getline(ss, token, ':');
    val.hours = oskar_settings_utility_string_to_int(token);
    if (val.hours < 0 || val.hours > 23) {
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
    bool ok = true;
    val.seconds = oskar_settings_utility_string_to_double(token, &ok);
    if (!ok) return false;
    if (val.seconds < 0.0 || val.seconds >= 60.0) {
        val.seconds = 0.0;
        return false;
    }
    return true;
}

static bool parse_date_style_1(const string& s, DateTime::Value& val)
{
//    cout << "FORMAT 1 = d-M-yyyy h:m:s[.z] : " << s << endl;
    istringstream ss(s);
    string token;
    getline(ss, token, '-');
    val.day = oskar_settings_utility_string_to_int(token);
    if (val.day < 1 || val.day > 31) {
        val.day = 0;
        return false;
    }
    getline(ss, token, '-');
    val.month = oskar_settings_utility_string_to_int(token);
    if (val.month < 1 || val.month > 12) {
        val.month = 0;
        return false;
    }
    getline(ss, token, ' ');
    val.year = oskar_settings_utility_string_to_int(token);
    getline(ss, token);
    return parse_time(token, val);
}

static bool parse_date_style_2(const string& s, DateTime::Value& val)
{
    //cout << "FORMAT 2 = yyyy/M/d/h:m:s[.z] : " << s << endl;
    istringstream ss(s);
    string token;
    getline(ss, token, '/');
    val.year = oskar_settings_utility_string_to_int(token);
    getline(ss, token, '/');
    val.month = oskar_settings_utility_string_to_int(token);
    if (val.month < 1 || val.month > 12) {
        val.month = 0;
        return false;
    }
    getline(ss, token, '/');
    val.day = oskar_settings_utility_string_to_int(token);
    if (val.day < 1 || val.day > 31) {
        val.day = 0;
        return false;
    }
    getline(ss, token);
    return parse_time(token, val);
}

static bool parse_date_style_3(const string& s, DateTime::Value& val)
{
//    cout << "FORMAT 3 = yyyy-M-d h:m:s[.z] : " << s << endl;
    istringstream ss(s);
    string token;
    getline(ss, token, '-');
    val.year = oskar_settings_utility_string_to_int(token);
    getline(ss, token, '-');
    val.month = oskar_settings_utility_string_to_int(token);
    if (val.month < 1 || val.month > 12) {
        val.month = 0;
        return false;
    }
    getline(ss, token, ' ');
    val.day = oskar_settings_utility_string_to_int(token);
    if (val.day < 1 || val.day > 31) {
        val.day = 0;
        return false;
    }
    getline(ss, token);
    return parse_time(token, val);
}

static bool parse_date_style_4(const string& s, DateTime::Value& val)
{
//    cout << "FORMAT 4 = yyyy-M-dTh:m:s[.z] : " << s << endl;
    istringstream ss(s);
    string token;
    getline(ss, token, '-');
    val.year = oskar_settings_utility_string_to_int(token);
    getline(ss, token, '-');
    val.month = oskar_settings_utility_string_to_int(token);
    if (val.month < 1 || val.month > 12) {
        val.month = 0;
        return false;
    }
    getline(ss, token, 'T');
    val.day = oskar_settings_utility_string_to_int(token);
    if (val.day < 1 || val.day > 31) {
        val.day = 0;
        return false;
    }
    getline(ss, token);
    return parse_time(token, val);
}

static DateTime::Value from_string(const string& s, bool& ok)
{
    DateTime::Value val;
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    double mjd = oskar_settings_utility_string_to_double(s, &ok);
    if (ok) {
        value_from_mjd(mjd, val);
        return val;
    }
    else {
        size_t p = s.find("-");
        if (p != string::npos)
        {
            if (p <= 2) {
                val.style = DateTime::BRITISH;
                ok = parse_date_style_1(s, val);
            }
            else if (s.find("T") != string::npos) {
                val.style = DateTime::ISO;
                ok = parse_date_style_4(s, val);
            }
            else {
                val.style = DateTime::INTERNATIONAL;
                ok = parse_date_style_3(s, val);
            }
        }
        else if (s.find("/") != string::npos) {
            val.style = DateTime::CASA;
            ok = parse_date_style_2(s, val);
        }
        return val;
    }
}

static string to_string(const DateTime::Value& val)
{
    ostringstream ss;
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    switch (val.style)
    {
        case DateTime::BRITISH:
        {
            ss << setfill('0') << setw(2) << val.day << '-';
            ss << setfill('0') << setw(2) << val.month << '-';
            ss << setfill('0') << setw(4) << val.year << ' ';
            ss << setfill('0') << setw(2) << val.hours << ':';
            ss << setfill('0') << setw(2) << val.minutes << ':';
            if (val.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
            break;
        }
        case DateTime::CASA:
        {
            ss << setfill('0') << setw(4) << val.year << '/';
            ss << setfill('0') << setw(2) << val.month << '/';
            ss << setfill('0') << setw(2) << val.day << '/';
            ss << setfill('0') << setw(2) << val.hours << ':';
            ss << setfill('0') << setw(2) << val.minutes << ':';
            if (val.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
            break;
        }
        case DateTime::INTERNATIONAL:
        {
            ss << setfill('0') << setw(4) << val.year << '-';
            ss << setfill('0') << setw(2) << val.month << '-';
            ss << setfill('0') << setw(2) << val.day << ' ';
            ss << setfill('0') << setw(2) << val.hours << ':';
            ss << setfill('0') << setw(2) << val.minutes << ':';
            if (val.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
            break;
        }
        case DateTime::ISO:
        {
            ss << setfill('0') << setw(4) << val.year << '-';
            ss << setfill('0') << setw(2) << val.month << '-';
            ss << setfill('0') << setw(2) << val.day << 'T';
            ss << setfill('0') << setw(2) << val.hours << ':';
            ss << setfill('0') << setw(2) << val.minutes << ':';
            if (val.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(val.seconds, 'f', 12);
            break;
        }
        case DateTime::MJD:
        {
            double mjd = value_to_mjd(val);
            ss << oskar_settings_utility_double_to_string_2(mjd, 'g', 14);
            break;
        }
        default:
            return string();
            break;
    }
    return ss.str();
}

DateTime::DateTime()
{
    (void) DateTime::init(0);
}

// LCOV_EXCL_START
DateTime::~DateTime()
{
}
// LCOV_EXCL_STOP

bool DateTime::init(const char* /*s*/)
{
    value_.clear();
    default_.clear();
    str_value_ = to_string(value_);
    str_default_ = to_string(default_);
    return true;
} // LCOV_EXCL_LINE

bool DateTime::set_default(const char* s)
{
    bool ok = true;
    default_ = from_string(s, ok);
    str_default_ = to_string(default_);
    if (ok) set_value(s); else (void) init(0);
    return ok;
} // LCOV_EXCL_LINE

bool DateTime::set_value(const char* s)
{
    bool ok = true;
    value_ = from_string(s, ok);
    str_value_ = to_string(value_);
    return ok;
} // LCOV_EXCL_LINE

bool DateTime::is_default() const
{
    bool answer = true;
    answer &= (value_.year == default_.year);
    answer &= (value_.month == default_.month);
    answer &= (value_.day == default_.day);
    answer &= (value_.hours == default_.hours);
    answer &= (value_.minutes == default_.minutes);
    answer &= (fabs(value_.seconds - default_.seconds) < 1e-6);
    return answer;
}

DateTime::Value DateTime::value() const
{
    return value_;
}

DateTime::Value DateTime::default_value() const
{
    return default_;
}

double DateTime::to_mjd() const
{
    return value_to_mjd(value_);
}

/**
 *  Convert a date to Julian Day.
 *
 *  Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
 *       4th ed., Duffet-Smith and Zwart, 2011.
 */
double DateTime::to_mjd_2() const
{
    int year = value_.year;
    int month = value_.month;
    int day = value_.day;
    int hours = value_.hours;
    int minutes = value_.minutes;
    double seconds = value_.seconds;

    int yearp = 0;
    int monthp = 0;
    if (month == 1 || month == 2)
    {
       yearp = year - 1;
       monthp = year + 12;
    }
    else
    {
        yearp = year;
        monthp = month;
    }
    int B = 0;
    if ((year < 1582) || (year == 1582 && month < 10) ||
                    (year == 1582 && month == 10 && day < 15))
    {
        B = 0;
    }
    else
    {
        int A = static_cast<int>(trunc(yearp / 100.));
        B = 2 - A + static_cast<int>(trunc(A / 4.));
    }
    int C = 0;
    if (yearp < 0)
    {
        C = static_cast<int>(trunc((365.25 * yearp) - 0.75));
    }
    else
    {
        C = static_cast<int>(trunc(365.25 * yearp));
    }
    int D = static_cast<int>(trunc(30.6001 * (monthp + 1)));
    double day_fraction = (hours + (minutes / 60.) + (seconds / 3600.)) / 24.;
    // double jd = (B + C + D + day) + 1720994.5 + day_fraction;
    double mjd = (B + C + D + day - 679006) + day_fraction;
    return mjd;
}

void DateTime::from_mjd(double mjd)
{
    value_from_mjd(mjd, value_);
}

DateTime::Format DateTime::format() const
{
    return value_.style;
}

bool DateTime::operator==(const DateTime& other) const
{
    bool answer = true;
    answer &= (value_.year == other.value_.year);
    answer &= (value_.month == other.value_.month);
    answer &= (value_.day == other.value_.day);
    answer &= (value_.hours == other.value_.hours);
    answer &= (value_.minutes == other.value_.minutes);
    answer &= (fabs(value_.seconds - other.value_.seconds) < 1e-6);
    return answer;
}

bool DateTime::operator>(const DateTime& other) const
{
    return this->to_mjd() > other.to_mjd();
}

} // namespace oskar
