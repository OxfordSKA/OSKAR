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
#include <cstring>
#include <sstream>
#include <iostream>

using namespace std;
using ttl::var::get;

namespace oskar {

bool compare(const IntListExt::Value& a, const IntListExt::Value& b)
{
    if (a.is_singular() || b.is_singular()) return false;
    if (a.which() != b.which()) return false;
    if (a.which() != 0)
        return get<string>(a) == get<string>(b);
    else
    {
        const vector<int>& v = get<vector<int> >(a);
        const vector<int>& d = get<vector<int> >(b);
        if (v.size() != d.size()) return false;
        for (size_t i = 0; i < v.size(); ++i) if (v[i] != d[i]) return false;
    }
    return true;
}

IntListExt::IntListExt() : delimiter_(',')
{
}

IntListExt::~IntListExt()
{
}

bool IntListExt::init(const char* s)
{
    if (!s || strlen(s) == 0) return false;
    special_value_ = string(s);
    default_ = special_value_;
    value_ = special_value_;
    return true;
}

bool IntListExt::set_default(const char* value)
{
    bool ok = from_string_(value, default_);
    str_default_ = to_string_(default_);
    if (ok)
        set_value(value);
    return ok;
}

bool IntListExt::set_value(const char* value)
{
    bool ok = from_string_(value, value_);
    str_value_ = to_string_(value_);
    return ok;
}

bool IntListExt::is_default() const
{
    return compare(value_, default_);
}

const char* IntListExt::special_string() const
{
    return special_value_.c_str();
}

bool IntListExt::is_extended() const
{
    return (!value_.is_singular() && value_.which() == 1) ? true : false;
}

int IntListExt::size() const
{
    if (value_.is_singular()) return 0;
    return (value_.which() == 0) ? (int) get<vector<int> >(value_).size() : 1;
}

const int* IntListExt::values() const
{
    if (!value_.is_singular() && value_.which() == 0)
        return (size() > 0) ? &get<vector<int> >(value_)[0] : 0;
    return 0;
}

bool IntListExt::operator==(const IntListExt& other) const
{
    return compare(value_, other.value_);
}

bool IntListExt::operator>(const IntListExt&) const
{
    return false;
}

bool IntListExt::from_string_(const string& s, Value& val) const
{
    if (s.find(delimiter_) == string::npos && s == special_value_)
    {
        val = s;
        return true;
    }

    // Convert the string to a vector of ints.
    vector<int> values;
    istringstream ss(s);
    string token;
    while (getline(ss, token, delimiter_))
    {
        bool valid = true;
        int v = oskar_settings_utility_string_to_int(token, &valid);
        if (!valid) return false;
        values.push_back(v);
    }
    val = values;
    return true;
}

string IntListExt::to_string_(const Value& v)
{
    if (v.is_singular()) return string();
    if (v.which() != 0) return get<string>(v);
    const vector<int>& values = get<vector<int> >(v);
    ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        ss << values.at(i);
        if (i < values.size() - 1) ss << delimiter_;
    }
    return ss.str();
}

} // namespace oskar
