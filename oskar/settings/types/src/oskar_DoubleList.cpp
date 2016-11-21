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
#include <oskar_DoubleList.hpp>
#include <cmath>
#include <iostream>

namespace oskar {

DoubleList::DoubleList()
: delimiter_(',')
{
}

DoubleList::~DoubleList()
{
}

bool DoubleList::init(const std::string& /*s*/)
{
    // TODO(BM) Set the delimiter from an initialisation string.
    delimiter_ = ',';
    return true;
}

bool DoubleList::set_default(const std::string& s)
{
    bool ok = from_string_(default_, s);
    if (ok) {
        value_ = default_;
    }
    return ok;
}

std::string DoubleList::get_default() const
{
    return to_string_(default_);
}

bool DoubleList::set_value(const std::string& s)
{
    return from_string_(value_, s);
}

std::string DoubleList::get_value() const
{
    return to_string_(value_);
}

bool DoubleList::is_default() const
{
    if (value_.size() != default_.size())
        return false;
    for (unsigned i = 0; i < value_.size(); ++i) {
        if (value_[i] != default_[i]) {
            return false;
        }
    }
    return true;
}

std::vector<double> DoubleList::values() const
{
    return value_;
}

bool DoubleList::from_string_(std::vector<double>& values,
                              const std::string& s) const
{
    // Clear any existing values.
    values.clear();

    // Convert the string to a vector of doubles.
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter_))
    {
        bool valid = true;
        double v = oskar_settings_utility_string_to_double(token, &valid);
        if (!valid) return false;
        values.push_back(v);
    }
    return true;
}

bool DoubleList::operator==(const DoubleList& other) const
{
    if (value_.size() != other.value_.size()) return false;
    for (unsigned i = 0; i < value_.size(); ++i)
        if (value_[i] != other.value_[i]) return false;
    return true;
}

bool DoubleList::operator>(const DoubleList& ) const
{
    return false;
}

std::string DoubleList::to_string_(const std::vector<double>& values) const
{
    std::ostringstream oss;
    for (size_t i = 0; i < values.size(); ++i) {
        double f = values.at(i);
        oss << oskar_settings_utility_double_to_string_2(f, 'g');
        if (i < values.size() - 1)
            oss << delimiter_;
    }
    return oss.str();
}



} // namespace oskar

