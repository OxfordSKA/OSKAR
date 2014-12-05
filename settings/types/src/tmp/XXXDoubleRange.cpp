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

#include <DoubleRange.hpp>

#include <cmath>
#include <cfloat>
#include <sstream>
#include <iostream>
#include <cstdlib>
#include <cerrno>
#include <cstring>

using namespace ttl::var;

namespace oskar {

DoubleRange::DoubleRange() : min_(-DBL_MAX), max_(DBL_MAX)
{
}

DoubleRange::DoubleRange(double min, double max, double value)
: min_(min), max_(max)
{
    set(value);
}

DoubleRange::DoubleRange(double min, double max, const std::string& ext_min)
: min_(min), max_(max), ext_min_(ext_min)
{}

DoubleRange::DoubleRange(double min, double max, const std::string& ext_min,
        const std::string& ext_max)
: min_(min), max_(max), ext_min_(ext_min), ext_max_(ext_max)
{}

bool DoubleRange::isSet() const
{
    return !(value_.is_singular());
}

std::string DoubleRange::toString(bool* ok) const
{
    // Value is defined and is a double.
    if (!value_.is_singular() && value_.which() == DOUBLE)
    {
        if (ok) *ok = true;
        std::ostringstream ss;
        ss << get<double>(value_);
        return ss.str();
    }
    else if (!value_.is_singular() && value_.which() == STD_STRING)
    {
        if (ok) *ok = true;
        return get<std::string>(value_);
    }
    else
    {
        if (ok) *ok = false;
        return std::string();
    }
}

void DoubleRange::set(const std::string& s, bool* ok)
{
    if (ok) *ok = true;

    if (s == ext_min_)
        value_ = std::string(ext_min_);
    else if (s == ext_max_)
        value_ = std::string(ext_max_);
    else {
        value_ = strtod(s.c_str(), 0);
        if (std::abs(get<double>(value_)) == HUGE_VAL) {
            if (ok) *ok = false;
#if 0
            std::cerr << "ERROR: [" << __PRETTY_FUNCTION__ << "] ";
            std::cerr << "Failed to convert string to double. ";
            std::cerr << strerror(errno) << std::endl;
#endif
            //value_ = 0.0;
        }
    }
}


void DoubleRange::set(double d, bool* ok)
{
    if (ok) *ok = true;

    // If in range, set to the specified value. Otherwise set to the value at
    // closest end of the range.
    if (d >= min_ && d <= max_) {
        value_ = static_cast<double>(d);
    }

    // Note: cast to long double to avoid over/underflow issues.
    else if (std::abs((long double)(d - min_)) < std::abs((long double)(max_-d))) {
        if (ext_min_.empty()) {
            if (ok) *ok = false;
            value_ = static_cast<double>(min_);
        }
        else {
            value_ = std::string(ext_min_);
        }
    }
    else {
        if (ext_max_.empty()) {
            if (ok) *ok = false;
            value_ = static_cast<double>(max_);
        }
        else {
            value_ = std::string(ext_max_);
        }
    }
}


double DoubleRange::getDouble(bool* ok) const
{
    if (!value_.is_singular() && value_.which() == DOUBLE)
    {
        if (ok) *ok = true;
        return get<double>(value_);
    }
    else if (!value_.is_singular() && value_.which() == STD_STRING)
    {
        std::string s = get<std::string>(value_);
        if (s == ext_min_)
        {
            if (ok) *ok = false;
            return (min_ == -DBL_MAX) ? min_ : min_-DBL_MIN;
        }
        else if (s == ext_max_)
        {
            if (ok) *ok = false;
            return (max_ == DBL_MAX) ? max_ : max_+DBL_MIN;
        }
        else {
            double value = strtod(s.c_str(), 0);
            if (std::abs(value) == HUGE_VAL) {
                if (ok) *ok = false;
#if 0
                std::cerr << "ERROR: " << __FUNCTION__;
                std::cerr << "failed to convert string to double. ";
                std::cerr << strerror(errno) << std::endl;
#endif
            }
            return value;
#if 0
            try
            {
                if (ok) *ok = true;
                return boost::lexical_cast<double>(get<std::string>(value_));
            }
            catch (boost::bad_lexical_cast&) {
                if (ok) *ok = false;
                return 0.0;
            }
#endif
        }
    }
    else
    {
        if (ok) *ok = false;
        return 0.0;
    }
}

} // namespace oskar

