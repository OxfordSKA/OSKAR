/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_settings_utility_string.h"

using std::string;

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
