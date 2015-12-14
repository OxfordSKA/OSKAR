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

#include <oskar_DateTime.hpp>
#include <oskar_settings_utility_string.hpp>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <cfloat>
#include <cmath>
#include <cstdio>

using namespace std;
namespace oskar {

DateTime::DateTime()
{
}

DateTime::~DateTime()
{
}

bool DateTime::init(const std::string& /*s*/)
{
    value_.clear();
    default_.clear();
    return true;
}


bool DateTime::set_default(const std::string& s)
{
    bool ok = true;
    default_ = string_to_date_time_(s, ok);
    if (ok) {
        value_ = default_;
    }
    return ok;
}

std::string DateTime::get_default() const
{
    return date_time_to_string(default_);
}

bool DateTime::set_value(const std::string& s)
{
//    cout << "DateTime::set_value() " << s << endl;
    bool ok = true;
    value_ = string_to_date_time_(s, ok);
    return ok;
}

std::string DateTime::get_value() const
{
    return date_time_to_string(value_);
}

bool DateTime::is_default() const
{
    bool answer = true;
    answer &= (value_.year == default_.year);
    answer &= (value_.month == default_.month);
    answer &= (value_.day == default_.day);
    answer &= (value_.hours == default_.hours);
    answer &= (value_.minutes == default_.minutes);
    answer &= (fabs(value_.seconds - default_.seconds) < DBL_MIN);
    return answer;
}

double DateTime::to_mjd() const
{
    // Compute Julian Day Number (Note: all integer division).
    int a = (14 - value_.month) / 12;
    int y = value_.year + 4800 - a;
    int m = value_.month + 12 * a - 3;
    int jdn = value_.day + (153 * m + 2) / 5 + (365 * y) + (y / 4) - (y / 100)
                            + (y / 400) - 32045;

    // Compute day fraction.
    double hours = value_.hours + value_.minutes / 60.0 +
                    value_.seconds / 3600.0;
    double day_fraction = hours / 24.0;
    day_fraction -= 0.5;
    return (jdn - 2400000.5) + day_fraction;
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

    int yearp;
    int monthp;
    if (month == 1 || month == 2) {
       yearp = year - 1;
       monthp = year + 12;
    }
    else {
        yearp = year;
        monthp = month;
    }
    int B;
    if ((year < 1582) || (year == 1582 && month < 10) ||
                    (year == 1582 && month == 10 && day < 15)) {
        B = 0;
    }
    else {
        int A = static_cast<int>(trunc(yearp / 100.));
        B = 2 - A + static_cast<int>(trunc(A / 4.));
    }
    int C;
    if (yearp < 0) {
        C = static_cast<int>(trunc((365.25 * yearp) - 0.75));
    } else {
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
    from_mjd_(mjd, value_);
}

bool DateTime::operator==(const DateTime& other) const
{
    bool answer = true;
    answer &= (value_.year == other.value_.year);
    answer &= (value_.month == other.value_.month);
    answer &= (value_.day == other.value_.day);
    answer &= (value_.hours == other.value_.hours);
    answer &= (value_.minutes == other.value_.minutes);
    answer &= (fabs(value_.seconds - other.value_.seconds) < DBL_MIN);
    return answer;
}

bool DateTime::operator>(const DateTime& other) const
{
    return this->to_mjd() > other.to_mjd();
}

/**
 * @brief
 * Convert Modified Julian Day to date.
 *
 * @details
 *  Algorithm from 'Practical Astronomy with your Calculator or Spreadsheet',
 *       4th ed., Duffet-Smith and Zwart, 2011.
 */
void DateTime::from_mjd_(double mjd, Value& value) const
{
    double iMJD;
    double F = modf(mjd, &iMJD);
    int I = static_cast<int>(iMJD) + 2400001;
    int A = static_cast<int>(trunc((I - 1867216.25) / 36524.25));
    int B;
    if (I > 2299160)
        B = I + 1 + A - static_cast<int>(trunc(A / 4.));
    else
        B = I;

    int C = B + 1524;
    int D = static_cast<int>(trunc((C - 122.1) / 365.25));
    int E = static_cast<int>(trunc(365.25 * D));
    int G = static_cast<int>(trunc((C - E) / 30.6001));
    double day = C - E + F - static_cast<int>(trunc(30.6001 * G));

    int month, year;
    if (G < 13.5)
        month = G - 1;
    else
        month = G - 13;

    if (month > 2.5)
        year = D - 4716;
    else
        year = D - 4715;

    double iday, fday;
    fday = modf(day, &iday);
    double hours = trunc(fday * 24.);
    double minutes = trunc(((fday * 24.) - hours) * 60.);
    double seconds = ((fday * 24.0) - hours - (minutes / 60.)) * 3600.0;
    value.year = static_cast<int>(year);
    value.month = static_cast<int>(month);
    value.day = iday;
    value.hours = static_cast<int>(hours);
    value.minutes = static_cast<int>(minutes);
    value.seconds = seconds;
    value.style = MJD;
}

DateTime::Value DateTime::string_to_date_time_(const std::string& s,
                                               bool& ok) const
{
    Value dateTime;
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    double mjd = oskar_settings_utility_string_to_double(s, &ok);
    if (ok) {
        from_mjd_(mjd, dateTime);
        return dateTime;
    }
    else {
        size_t p = s.find("-");
        if (p != std::string::npos)
        {
            if (p <= 2) {
                dateTime.style = BRITISH;
                ok = parse_date_style_1_(s, dateTime);
            }
            else if (s.find("T") != std::string::npos) {
                dateTime.style = ISO;
                ok = parse_date_style_4_(s, dateTime);
            }
            else {
                dateTime.style = INTERNATIONAL;
                ok = parse_date_style_3_(s, dateTime);
            }
        }
        else if (s.find("/") != std::string::npos) {
            dateTime.style = CASA;
            ok = parse_date_style_2_(s, dateTime);
        }
        return dateTime;
    }
}

std::string DateTime::date_time_to_string(const Value& dateTime) const
{
    std::ostringstream ss;
    /*
     *  1. d-M-yyyy h:m:s[.z] - British style
     *  2. yyyy/M/d/h:m:s[.z] - CASA style
     *  3. yyyy-M-d h:m:s[.z] - International style
     *  4. yyyy-M-dTh:m:s[.z] - ISO date style
     *  5. MJD
     */
    switch (dateTime.style)
    {
        case BRITISH:
        {
            ss << std::setfill('0') << std::setw(2) << dateTime.day << '-';
            ss << std::setfill('0') << std::setw(2) << dateTime.month << '-';
            ss << std::setfill('0') << std::setw(4) << dateTime.year << ' ';
            ss << std::setfill('0') << std::setw(2) << dateTime.hours << ':';
            ss << std::setfill('0') << std::setw(2) << dateTime.minutes << ':';
            if (dateTime.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(dateTime.seconds, 'f', 12);
            break;
        }
        case CASA:
        {
            ss << std::setfill('0') << std::setw(4) << dateTime.year << '/';
            ss << std::setfill('0') << std::setw(2) << dateTime.month << '/';
            ss << std::setfill('0') << std::setw(2) << dateTime.day << '/';
            ss << std::setfill('0') << std::setw(2) << dateTime.hours << ':';
            ss << std::setfill('0') << std::setw(2) << dateTime.minutes << ':';
            if (dateTime.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(dateTime.seconds, 'f', 12);
            break;
        }
        case INTERNATIONAL:
        {
            ss << std::setfill('0') << std::setw(4) << dateTime.year << '-';
            ss << std::setfill('0') << std::setw(2) << dateTime.month << '-';
            ss << std::setfill('0') << std::setw(2) << dateTime.day << ' ';
            ss << std::setfill('0') << std::setw(2) << dateTime.hours << ':';
            ss << std::setfill('0') << std::setw(2) << dateTime.minutes << ':';
            if (dateTime.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(dateTime.seconds, 'f', 12);
            break;
        }
        case ISO:
        {
            ss << std::setfill('0') << std::setw(4) << dateTime.year << '-';
            ss << std::setfill('0') << std::setw(2) << dateTime.month << '-';
            ss << std::setfill('0') << std::setw(2) << dateTime.day << 'T';
            ss << std::setfill('0') << std::setw(2) << dateTime.hours << ':';
            ss << std::setfill('0') << std::setw(2) << dateTime.minutes << ':';
            if (dateTime.seconds < 10.) ss << 0;
            ss << oskar_settings_utility_double_to_string_2(dateTime.seconds, 'f', 12);
            break;
        }
        case MJD:
        {
            double mjd = to_mjd();
            ss << oskar_settings_utility_double_to_string_2(mjd, 'g', 14);
            break;
        }
        default:
            return std::string();
            break;
    }
    return ss.str();
}

bool DateTime::parse_date_style_1_(const std::string& s, Value& dateTime) const
{
//    std::cout << "FORMAT 1 = d-M-yyyy h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    dateTime.day = oskar_settings_utility_string_to_int(token);
    if (dateTime.day < 0 || dateTime.day >= 31) {
        dateTime.day = 0;
        return false;
    }
    std::getline(ss, token, '-');
    dateTime.month = oskar_settings_utility_string_to_int(token);
    if (dateTime.month < 0 || dateTime.month >= 12) {
        dateTime.month = 0;
        return false;
    }
    std::getline(ss, token, ' ');
    dateTime.year = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token);
    return parse_time_(token, dateTime);
}

bool DateTime::parse_date_style_2_(const std::string& s, Value& dateTime) const
{
    //std::cout << "FORMAT 2 = yyyy/M/d/h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '/');
    dateTime.year = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '/');
    dateTime.month = oskar_settings_utility_string_to_int(token);
    if (dateTime.month < 0 || dateTime.month >= 12) {
        dateTime.month = 0;
        return false;
    }
    std::getline(ss, token, '/');
    dateTime.day = oskar_settings_utility_string_to_int(token);
    if (dateTime.day < 0 || dateTime.day >= 31) {
        dateTime.day = 0;
        return false;
    }
    std::getline(ss, token);
    return parse_time_(token, dateTime);
}

bool DateTime::parse_date_style_3_(const std::string& s, Value& dateTime) const
{
//    std::cout << "FORMAT 3 = yyyy-M-d h:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    dateTime.year = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '-');
    dateTime.month = oskar_settings_utility_string_to_int(token);
    if (dateTime.month < 0 || dateTime.month >= 12) {
        dateTime.month = 0;
        return false;
    }
    std::getline(ss, token, ' ');
    dateTime.day = oskar_settings_utility_string_to_int(token);
    if (dateTime.day < 0 || dateTime.day >= 31) {
        dateTime.day = 0;
        return false;
    }
    std::getline(ss, token);
    return parse_time_(token, dateTime);
}

bool DateTime::parse_date_style_4_(const std::string& s, Value& dateTime) const
{
//    std::cout << "FORMAT 4 = yyyy-M-dTh:m:s[.z] : " << s << std::endl;
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, '-');
    dateTime.year = oskar_settings_utility_string_to_int(token);
    std::getline(ss, token, '-');
    dateTime.month = oskar_settings_utility_string_to_int(token);
    if (dateTime.month < 0 || dateTime.month >= 12) {
        dateTime.month = 0;
        return false;
    }
    std::getline(ss, token, 'T');
    dateTime.day = oskar_settings_utility_string_to_int(token);
    if (dateTime.day < 0 || dateTime.day >= 31) {
        dateTime.day = 0;
        return false;
    }
    std::getline(ss, token);
    return parse_time_(token, dateTime);
}

bool DateTime::parse_time_(const std::string& s, Value& dateTime) const
{
    std::istringstream ss(s);
    std::string token;
    std::getline(ss, token, ':');
    dateTime.hours = oskar_settings_utility_string_to_int(token);
    if (dateTime.hours < 0 || dateTime.hours >= 23) {
        dateTime.hours = 0;
        return false;
    }
    std::getline(ss, token, ':');
    dateTime.minutes = oskar_settings_utility_string_to_int(token);
    if (dateTime.minutes < 0 || dateTime.minutes >= 59) {
        dateTime.minutes = 0;
        return false;
    }
    std::getline(ss, token);
    bool ok = true;
    dateTime.seconds = oskar_settings_utility_string_to_double(token, &ok);
    if (!ok) return false;
    if (dateTime.minutes < 0.0 || dateTime.minutes >= 60.0) {
        dateTime.seconds = 0.0;
        return false;
    }
    return true;
}


} // namespace oskar

