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


#include <oskar_SettingsKey.hpp>
#include <oskar_settings_utility_string.hpp>
#include <iostream>

using namespace std;

namespace oskar {

SettingsKey::SettingsKey(char separator)
: sep_(separator)
{
}

SettingsKey::SettingsKey(const std::string& key, char separator)
: key_(key), sep_(separator)
{
    tokens_.clear();
    std::string s = key_;
    std::string delimiter(1, sep_);
    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens_.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens_.push_back(s);
}

SettingsKey::~SettingsKey()
{
}

SettingsKey::SettingsKey(const SettingsKey& other)
{
    key_ = other.key_;
    tokens_ = other.tokens_;
    sep_ = other.sep_;
}


void SettingsKey::set_separator(char s)
{
    sep_ = s;
}

char SettingsKey::separator() const
{
    return sep_;
}

bool SettingsKey::empty() const
{
    return key_.empty();
}

bool SettingsKey::operator==(const SettingsKey& other) const
{
    return (oskar_settings_utility_string_to_upper(key_) ==
                    oskar_settings_utility_string_to_upper(other.key_));
}

SettingsKey::operator std::string() const
{
    return key_;
}

std::string SettingsKey::operator[](int i) const
{
    return tokens_[i];
}

int SettingsKey::depth() const
{
    return tokens_.size() - 1;
}

int SettingsKey::size() const
{
    return tokens_.size();
}

std::string SettingsKey::front() const
{
    return tokens_.front();
}

std::string SettingsKey::back() const
{
    return tokens_.back();
}

std::string SettingsKey::group() const
{
    std::string s;
    for (unsigned i = 0; i < tokens_.size() - 1; ++i) {
        s += tokens_[i] + sep_;
    }
    return s;
}

const char* SettingsKey::c_str() const
{
    return key_.c_str();
}

std::ostream& operator<< (std::ostream& stream, const SettingsKey& k)
{
    return stream << k.key_;
}

} // namespace oskar
