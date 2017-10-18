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
#include "settings/types/oskar_StringList.h"

#include <sstream>

using namespace std;

namespace oskar {

static string to_string(const vector<string>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0u; i < values.size(); ++i)
    {
        ss << values.at(i);
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

StringList::StringList() : delimiter_(',')
{
}

StringList::~StringList()
{
}

bool StringList::init(const char* /*s*/)
{
     // TODO(BM) allow a a different delimiter via the init method?
     value_.clear();
     default_.clear();
     pointers_.clear();
     return true;
}

bool StringList::set_default(const char* s)
{
    default_ = oskar_settings_utility_string_get_type_params(s);
    str_default_ = to_string(default_, delimiter_);
    set_value(s);
    return true;
}

bool StringList::set_value(const char* s)
{
    value_ = oskar_settings_utility_string_get_type_params(s);
    str_value_ = to_string(value_, delimiter_);
    pointers_.clear();
    for (size_t i = 0; i < value_.size(); ++i)
        pointers_.push_back(value_[i].c_str());
    return true;
}

bool StringList::is_default() const
{
    return compare_vectors(value_, default_);
}

int StringList::size() const { return (int) value_.size(); }

const char* const* StringList::values() const
{
    return pointers_.size() > 0 ? &pointers_[0] : 0;
}

bool StringList::operator==(const StringList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool StringList::operator>(const StringList&) const
{
    return false;
}

} // namespace oskar
