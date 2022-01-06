/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_RandomSeed.h"

namespace oskar {

static bool from_string(const char* s, int& value)
{
    if (oskar_settings_utility_string_starts_with("TIME", s, false)) {
        value = -1;
        return true;
    }
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;
    if (i < 1) return false;
    value = i;
    return true;
}

static std::string to_string(int value)
{
    return (value < 1) ? "time" : oskar_settings_utility_int_to_string(value);
}

RandomSeed::RandomSeed()
{
    (void) RandomSeed::init(0);
}

// LCOV_EXCL_START
RandomSeed::~RandomSeed()
{
}
// LCOV_EXCL_STOP

bool RandomSeed::init(const char* /*s*/)
{
    default_ = 1;
    value_ = 1;
    str_default_ = "1";
    str_value_ = "1";
    return true;
}

bool RandomSeed::set_default(const char* s)
{
    bool ok = from_string(s, default_);
    str_default_ = to_string(default_);
    if (ok) set_value(s); else init(0);
    return ok;
} // LCOV_EXCL_LINE

bool RandomSeed::set_value(const char* s)
{
    bool ok = from_string(s, value_);
    str_value_ = to_string(value_);
    return ok;
} // LCOV_EXCL_LINE

bool RandomSeed::is_default() const
{
    return value_ == default_;
}

int RandomSeed::value() const
{
    return value_;
}

int RandomSeed::default_value() const
{
    return default_;
}

bool RandomSeed::operator==(const RandomSeed& other) const
{
    return value_ == other.value_;
}

bool RandomSeed::operator>(const RandomSeed& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
