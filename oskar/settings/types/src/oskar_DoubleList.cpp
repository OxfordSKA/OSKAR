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
#include "settings/types/oskar_DoubleList.h"
#include <sstream>
#include <cmath>
#include <iostream>

using namespace std;

namespace oskar {

static bool from_string(vector<double>& values, const string& s, char delimiter)
{
    // Convert the string to a vector of doubles.
    vector<double> temp;
    istringstream ss(s);
    string token;
    while (getline(ss, token, delimiter))
    {
        bool valid = true;
        double v = oskar_settings_utility_string_to_double(token, &valid);
        if (!valid) return false;
        temp.push_back(v);
    }
    values = temp;
    return true;
}

static string to_string(const vector<double>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        ss << oskar_settings_utility_double_to_string_2(values[i], 'g');
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

DoubleList::DoubleList()
{
    (void) init(0);
}

DoubleList::~DoubleList()
{
}

bool DoubleList::init(const char* /*s*/)
{
    // TODO(BM) Set the delimiter from an initialisation string.
    delimiter_ = ',';
    return true;
}

bool DoubleList::set_default(const char* s)
{
    bool ok = from_string(default_, s, delimiter_);
    str_default_ = to_string(default_, delimiter_);
    if (ok)
        set_value(s);
    return ok;
}

bool DoubleList::set_value(const char* s)
{
    bool ok = from_string(value_, s, delimiter_);
    str_value_ = to_string(value_, delimiter_);
    return ok;
}

bool DoubleList::is_default() const
{
    return compare_vectors(value_, default_);
}

int DoubleList::size() const
{
    return (int) value_.size();
}

const double* DoubleList::values() const
{
    return value_.size() > 0 ? &value_[0] : 0;
}

bool DoubleList::operator==(const DoubleList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool DoubleList::operator>(const DoubleList&) const
{
    return false;
}

} // namespace oskar
