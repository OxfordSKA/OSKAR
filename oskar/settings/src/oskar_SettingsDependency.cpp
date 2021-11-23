/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsKey.h"
#include "settings/oskar_SettingsDependency.h"
#include <cstring>

namespace oskar {

SettingsDependency::SettingsDependency(const char* key,
                       const char* value,
                       const char* logic)
{
    key_ = std::string(key);
    value_ = std::string(value);
    logic_ = SettingsDependency::string_to_logic(logic);
}

const char* SettingsDependency::key() const
{
    return key_.c_str();
}

const char* SettingsDependency::value() const
{
    return value_.c_str();
}

SettingsDependency::Logic SettingsDependency::logic() const
{
    return logic_;
}

bool SettingsDependency::is_valid() const
{
    if (logic_ == UNDEF || key_.empty() || value_.empty())
    {
        return false;
    }
    return true;
}

const char* SettingsDependency::logic_string() const
{
    return SettingsDependency::logic_to_string(logic_);
}

const char* SettingsDependency::logic_to_string(
        const SettingsDependency::Logic& v)
{
    switch (v) {
        case EQ: return "EQ";
        case NE: return "NE";
        case GT: return "GT";
        case GE: return "GE";
        case LT: return "LT";
        case LE: return "LE";
        default: return "";
    };
    return "";
}

SettingsDependency::Logic SettingsDependency::string_to_logic(const char* s)
{
    if (!s || strlen(s) == 0) return EQ;
    if (!strncmp(s, "EQ", 2)) return EQ;
    if (!strncmp(s, "NE", 2)) return NE;
    if (!strncmp(s, "GT", 2)) return GT;
    if (!strncmp(s, "GE", 2)) return GE;
    if (!strncmp(s, "LT", 2)) return LT;
    if (!strncmp(s, "LE", 2)) return LE;
    return UNDEF;
}

} // namespace oskar
