/*
 * Copyright (c) 2015-2017, The University of Oxford
 * All rights reserved.
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

#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_settings_utility_string.h"

using namespace std;

namespace oskar {

SettingsKey::SettingsKey(char separator) : sep_(separator) {}

SettingsKey::SettingsKey(const char* key, char separator)
{
    from_string(key, separator);
}

SettingsKey::~SettingsKey()
{
}

const char* SettingsKey::back() const { return tokens_.back().c_str(); }

int SettingsKey::depth() const { return (int) tokens_.size() - 1; }

bool SettingsKey::empty() const { return key_.empty(); }

void SettingsKey::from_string(const char* key, char separator)
{
    tokens_.clear();
    key_ = string(key);
    sep_ = separator;
    string s = key_;
    string delimiter(1, sep_);
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos)
    {
        token = s.substr(0, pos);
        tokens_.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens_.push_back(s);
}

char SettingsKey::separator() const { return sep_; }

void SettingsKey::set_separator(char s) { sep_ = s; }

bool SettingsKey::operator==(const SettingsKey& other) const
{
    return (oskar_settings_utility_string_to_upper(key_) ==
                    oskar_settings_utility_string_to_upper(other.key_));
}

const char* SettingsKey::operator[](int i) const { return tokens_[i].c_str(); }

SettingsKey::operator const char*() const { return key_.c_str(); }

} // namespace oskar
