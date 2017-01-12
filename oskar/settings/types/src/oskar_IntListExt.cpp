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

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_IntListExt.h"
#include <sstream>
#include <iostream>

using namespace std;

namespace oskar {

IntListExt::IntListExt()
: delimiter_(',')
{
}

IntListExt::~IntListExt()
{
}

bool IntListExt::init(const std::string& s)
{
    if (!s.empty()) {
        special_value_ = s;
        default_ = special_string();
        value_ = special_string();
        return true;
    }
    return false;
}

bool IntListExt::set_default(const std::string& value)
{
    if (value.find(delimiter_) == std::string::npos) {
        if (value == special_value_) {
            default_ = value;
            value_ = value;
            return true;
        }
        else {
            return false;
        }
    }
    else {
        std::vector<int> v;
        default_ = v;
        bool ok = from_string_(value, ttl::var::get<std::vector<int> >(default_));
        value_ = default_;
        return (ok && from_string_(value, v));
    }
}

std::string IntListExt::get_default() const
{
    if (!default_.is_singular()) {
        if (default_.which() == 0) {
            return to_string_(ttl::var::get<std::vector<int> >(default_));
        }
        else {
            return ttl::var::get<std::string>(default_);
        }
    }
    return std::string();
}

bool IntListExt::set_value(const std::string& value)
{
    if (value.find(delimiter_) == std::string::npos) {
        if (value == special_value_) {
            value_ = value;
            return true;
        } else {
            std::vector<int> v;
            value_ = v;
            return from_string_(value, ttl::var::get<std::vector<int> >(value_));
        }
    }
    else {
        std::vector<int> v;
        value_ = v;
        return from_string_(value, ttl::var::get<std::vector<int> >(value_));
    }
}

std::string IntListExt::get_value() const
{
    if (!value_.is_singular()) {
        if (value_.which() == 0) {
            return to_string_(ttl::var::get<std::vector<int> >(value_));
        }
        else {
            return ttl::var::get<std::string>(value_);
        }
    }
    return std::string();
}

bool IntListExt::is_default() const
{
    if (default_.is_singular() || value_.is_singular())
        return false;
    if (default_.which() == value_.which()) {
        if (default_.which() == 0) {
            const std::vector<int>& v = ttl::var::get<std::vector<int> >(value_);
            const std::vector<int>& d = ttl::var::get<std::vector<int> >(default_);
            if (v.size() != d.size()) {
                return false;
            }
            for (unsigned int i = 0; i < v.size(); ++i) {
                if (v.at(i) != d.at(i)) {
                    return false;
                }
            }
            return true;
        }
        else {
            return ttl::var::get<std::string>(default_) ==
                            ttl::var::get<std::string>(value_);
        }
    }
    return false;
}

bool IntListExt::is_extended() const
{
    if (!value_.is_singular() && value_.which() == 1) {
        return true;
    }
    return false;
}

int IntListExt::size() const
{
    if (!value_.is_singular()) {
        if (value_.which() == 0) {
            return ttl::var::get<std::vector<int> >(value_).size();
        }
        else {
            return 1;
        }
    }
    return 0;
}

std::vector<int> IntListExt::values() const
{
    if (!value_.is_singular() && value_.which() == 0) {
        return ttl::var::get<std::vector<int> >(value_);
    }
    return std::vector<int>();
}

bool IntListExt::operator==(const IntListExt& other) const
{
    using ttl::var::get;

    if (value_.is_singular() || other.value_.is_singular())
        return false;
    if (value_.which() == other.value_.which()) {
        if (default_.which() == 0) {
            const std::vector<int>& v = get<std::vector<int> >(value_);
            const std::vector<int>& d = get<std::vector<int> >(other.value_);
            if (v.size() != d.size()) {
                return false;
            }
            for (unsigned int i = 0; i < v.size(); ++i) {
                if (v.at(i) != d.at(i)) {
                    return false;
                }
            }
            return true;
        }
        else {
            return get<std::string>(value_) == get<std::string>(other.value_);
        }
    }
    return false;
}

bool IntListExt::operator>(const IntListExt& ) const
{
    return false;
}

bool IntListExt::from_string_(const std::string& s, std::vector<int>& values) const
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

std::string IntListExt::to_string_(const std::vector<int>& values) const
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

