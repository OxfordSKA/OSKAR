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

#include <oskar_settings_utility_string.hpp>

#include <sstream>
#include <iomanip>
#include <climits>
#include <cstdlib>
#include <cerrno>
#include <cmath>
#include <stdarg.h>

#include <iostream> // for debugging FIXME
#include <iomanip>


std::string oskar_settings_utility_string_trim(const std::string& s,
        const std::string& whitespace)
{
    // Find the first index that does not match whitespace.
    size_t i0 = s.find_first_not_of(whitespace);
    if (i0 == std::string::npos) return "";

    // Find the last index that does not match whitespace.
    size_t i1 = s.find_last_not_of(whitespace);

    // Find the string with whitespace subtracted from each end.
    return s.substr(i0, i1 - i0 + 1);
}

std::vector<std::string> oskar_settings_utility_string_get_type_params(
        const std::string& s)
{
    std::vector<std::string> params;
    std::stringstream ss_all(s);
    std::string p;
    while (std::getline(ss_all, p, '"')) {
        std::stringstream ss(p);
        while (std::getline(ss, p, ',')) {
            p = oskar_settings_utility_string_trim(p);
            if (!p.empty()) params.push_back(p);
        }
        if (std::getline(ss_all, p, '"')) {
            if (!p.empty()) params.push_back(p);
        }
    }
    return params;
}

int oskar_settings_utility_string_to_int(const std::string& s, bool *ok)
{
    int base = 10;
    char *endptr;

    errno = 0;  // To distinguish success/failure after call
    long int val = strtol(s.c_str(), &endptr, base);

    // If argument ok is not null, check for various possible errors.
    if (ok) {
        *ok = true;
        //  Check for various possible errors
        if ((errno == ERANGE && (val == LONG_MAX || val == LONG_MIN))
                || (errno != 0 && val == 0))
            *ok = false;
        // No digits found.
        if (endptr == s.c_str()) *ok = false;
        // Further characters found in the string
        // NOTE this may not actually be an error ...
        if (*endptr != '\0') *ok = false;
    }

    // Value not in the integer range, set to 0 set fail flag.
    if (val > INT_MAX || val < -INT_MAX) {
        val = 0L;
        if (ok) *ok = false;
    }

    return static_cast<int>(val);
}

std::string oskar_settings_utility_int_to_string(int i)
{
    std::ostringstream ss;
    ss << i;
    return ss.str();
}

std::string oskar_settings_utility_string_to_upper(const std::string& s)
{
    std::string s_(s);
    for (size_t i = 0; i < s_.length(); ++i) s_[i] = toupper(s_[i]);
    return s_;
}

bool oskar_settings_utility_string_starts_with(const std::string& s1,
        const std::string& s2, bool case_senstive)
{
    std::string s1_(s1), s2_(s2);
    if (case_senstive == false) {
        s1_ = oskar_settings_utility_string_to_upper(s1);
        s2_ = oskar_settings_utility_string_to_upper(s2);
    }
    if (s1_.find(s2_) == 0) return true;
    else return false;
}


std::string oskar_settings_utility_double_to_string(double d,
        int precision /* = -17 */)
{
    std::ostringstream ss;

//    ss.setf(std::ios_base::fmtflags(), std::ios_base::floatfield);
//    ss.setf(std::ios_base::fixed, std::ios_base::floatfield);
//    ss.setf(std::ios_base::scientific, std::ios_base::floatfield);

    if (precision > 0) {
        ss.setf(std::ios_base::fixed);
        ss << std::setprecision(precision);
    }

    else if (precision < 0)
    {
        // Attempt to guess the number of decimal digits
//        int count = 0;
//        double num = std::abs(d);
//        double prev = num;
//        std::cout << "init: " << std::setprecision(12) << prev << std::endl;
//        num -= int(num);
//        while (std::abs(num) >= 1.0e-8 && count < 17) {
//            num *= 10;
//            num -= round(num);
//            //num -= int(num);
//            count++;
//            std::cout << prev << " " << count << " " << num << std::endl;
//            prev = num;
//        }
        //ss.setf(std::ios_base::fixed);
        //ss << std::setprecision(std::min(count, precision*-1));
        ss << std::setprecision(-precision);
    }

    ss << d;
    std::string s = std::string(ss.str());

//    if (strip_trailing_zeros) {
//
//    }

    return s;
}

double oskar_settings_utility_string_to_double(const std::string& s, bool *ok)
{
    char *endptr;

    errno = 0;  // To distinguish success/failure after call
    double val = strtod(s.c_str(), &endptr);

    // If argument ok is not null, check for various possible errors.
    if (ok) {
        *ok = true;
        //  Check for various possible errors
        if ((errno != 0 && val == 0.0) ||
                (errno == ERANGE && (val == HUGE_VAL || val == -HUGE_VAL))) {
            *ok = false;
        }
        // No digits found.
        if (endptr == s.c_str()) *ok = false;
        // Further characters found in the string
        // NOTE this may not actually be an error ...
        if (*endptr != '\0') *ok = false;
    }

    return val;
}

// http://stackoverflow.com/questions/2342162/stdstring-formatting-like-sprintf
std::string oskar_format_string(const std::string fmt, ...)
{
    int size = 512;
    char* buffer = 0;
    buffer = new char[size];
    va_list vl;
    va_start(vl, fmt);
    int nsize = vsnprintf(buffer, size, fmt.c_str(), vl);
    if(size <= nsize) { // fail? delete buffer and try again
        delete[] buffer;
        buffer = 0;
        buffer = new char[nsize+1]; //+1 for /0
        nsize = vsnprintf(buffer, size, fmt.c_str(), vl);
    }
    std::string ret(buffer);
    va_end(vl);
    delete[] buffer;
    return ret;
}
