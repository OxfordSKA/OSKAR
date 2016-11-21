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

namespace oskar {

SettingsKey::SettingsKey(char separator)
: sep_(separator)
{
}

SettingsKey::SettingsKey(const std::string& key, char separator)
: key_(key), sep_(separator)
{
}

SettingsKey::~SettingsKey()
{
}

std::string SettingsKey::toString() const
{
    return key_;
}

void SettingsKey::setSeparator(char s)
{
    sep_ = s;
}

char SettingsKey::separator() const
{
    return sep_;
}

bool SettingsKey::operator==(const SettingsKey& other) const
{
    return (key_ == other.key_);
}


SettingsKey::operator std::string() const
{
    return key_;
}

SettingsKey::operator const char*() const
{
    return key_.c_str();
}


int SettingsKey::depth() const
{
    return this->tokens().size()-1;
}

std::vector<std::string> SettingsKey::tokens() const
{
    std::vector<std::string> tokens;

    std::string s = key_;
    std::string delimiter(1, sep_);

    size_t pos = 0;
    std::string token;
    while ((pos = s.find(delimiter)) != std::string::npos) {
        token = s.substr(0, pos);
        tokens.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    tokens.push_back(s);

    return tokens;
}

std::string SettingsKey::leaf() const
{
    return this->tokens().back();
}

} // namespace oskar
