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
#include "settings/types/oskar_OptionList.h"

namespace oskar {

OptionList::OptionList()
{
}

OptionList::~OptionList()
{
}

bool OptionList::init(const std::string& s)
{
    options_.clear();
    options_ = oskar_settings_utility_string_get_type_params(s);
    return true;
}

bool OptionList::set_default(const std::string& value)
{
    bool ok = from_string_(default_, value);
    if (ok) {
        value_ = default_;
    }
    else {
        value_.clear();
        default_.clear();
    }
    return ok;
}

std::string OptionList::get_default() const
{
    return default_;
}

bool OptionList::set_value(const std::string& value)
{
    return from_string_(value_, value);
}

std::string OptionList::get_value() const
{
    return value_;
}

bool OptionList::is_default() const
{
    return value_ == default_;
}

bool OptionList::operator==(const OptionList& other) const
{
    return value_ == other.value_;
}

bool OptionList::operator>(const OptionList& ) const
{
    return false;
}

bool OptionList::from_string_(std::string& value, const std::string& s) const
{
    if (s.empty() && !default_.empty()) {
        return false;
    }
    if (default_.empty() && s.empty()) {
        value = s;
        return true;
    }
    for (size_t i = 0; i < options_.size(); ++i) {
        if (oskar_settings_utility_string_starts_with(options_[i], s)) {
            value = options_[i];
            return true;
        }
    }
    return false;
}

} // namespace oskar

