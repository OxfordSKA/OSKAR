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

#include "oskar_settings_utility_string.hpp"
#include <sstream>
#include "oskar_IntList.hpp"

namespace oskar {

IntList::IntList()
: delimiter_(',')
{
}

IntList::~IntList()
{
}

bool IntList::init(const std::string& /*s*/)
{
    // TODO(BM) Could use this to set the delimiter ... ?
    return true;
}

bool IntList::set_default(const std::string& value)
{
    default_.clear();
    value_.clear();
    bool ok = from_string_(value, default_);
    if (ok) {
        for (unsigned int i = 0; i < default_.size(); ++i) {
            value_.push_back(default_[i]);
        }
    }
    return ok;
}

std::string IntList::get_default() const
{
    return to_string_(default_);
}

bool IntList::set_value(const std::string& value)
{
    return from_string_(value, value_);
}

std::string IntList::get_value() const
{
    return to_string_(value_);
}

bool IntList::is_default() const
{
    if (value_.size() != default_.size())
        return false;
    for (unsigned int i = 0; i < value_.size(); ++i) {
        if (value_[i] != default_[i]) {
            return false;
        }
    }
    return true;
}

bool IntList::operator==(const IntList& other) const
{
    if (value_.size() != other.value_.size()) return false;
    for (unsigned int i = 0; i < value_.size(); ++i)
        if (value_[i] != other.value_[i]) return false;
    return true;
}

bool IntList::operator>(const IntList& ) const
{
    return false;
}

bool IntList::from_string_(const std::string& s, std::vector<int>& values) const
{
    // Clear any existing values.
    values.clear();

    // Convert the string to a vector of ints.
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter_))
    {
        bool valid = true;
        int v = oskar_settings_utility_string_to_int(token, &valid);
        if (!valid) {
            values.clear();
            return false;
        }
        values.push_back(v);
    }
    return true;
}

std::string IntList::to_string_(const std::vector<int>& values) const
{
    std::ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i) {
        ss << values.at(i);
        if (i < values.size() - 1)
            ss << delimiter_;
    }
    return ss.str();
}

} // namespace oskar

