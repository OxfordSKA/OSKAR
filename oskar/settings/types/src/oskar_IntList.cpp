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
#include "settings/types/oskar_IntList.h"
#include <sstream>

using namespace std;

namespace oskar {

static bool from_string(const string& s, vector<int>& values, char delimiter)
{
    // Convert the string to a vector of ints.
    vector<int> temp;
    istringstream ss(s);
    string token;
    while (getline(ss, token, delimiter))
    {
        bool valid = true;
        int v = oskar_settings_utility_string_to_int(token, &valid);
        if (!valid) return false;
        temp.push_back(v);
    }
    values = temp;
    return true;
}

static string to_string(const vector<int>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        ss << values.at(i);
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

IntList::IntList()
{
    (void) init(0);
}

IntList::~IntList()
{
}

bool IntList::init(const char* /*s*/)
{
    // TODO(BM) Could use this to set the delimiter ... ?
    delimiter_ = ',';
    return true;
}

bool IntList::set_default(const char* value)
{
    bool ok = from_string(value, default_, delimiter_);
    str_default_ = to_string(default_, delimiter_);
    if (ok)
        set_value(value);
    return ok;
}

bool IntList::set_value(const char* value)
{
    bool ok = from_string(value, value_, delimiter_);
    str_value_ = to_string(value_, delimiter_);
    return ok;
}

bool IntList::is_default() const
{
    return compare_vectors(value_, default_);
}

int IntList::size() const
{
    return (int) value_.size();
}

const int* IntList::values() const
{
    return value_.size() > 0 ? &value_[0] : 0;
}

bool IntList::operator==(const IntList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool IntList::operator>(const IntList&) const
{
    return false;
}

} // namespace oskar
