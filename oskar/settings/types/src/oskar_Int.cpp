/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_Int.h"
#include "settings/oskar_settings_utility_string.h"

namespace oskar {

Int::Int()
{
    (void) Int::init(0);
}

// LCOV_EXCL_START
Int::~Int()
{
}
// LCOV_EXCL_STOP

bool Int::init(const char* /*s*/)
{
    default_ = 0;
    value_ = 0;
    str_default_ = "0";
    str_value_ = "0";
    return true;
}

bool Int::set_default(const char* value)
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(value, &ok);
    if (!ok) return false;
    default_ = i;
    str_default_ = oskar_settings_utility_int_to_string(default_);
    this->set_value(value);
    return true;
}

bool Int::set_value(const char* value)
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(value, &ok);
    if (!ok) return false;
    value_ = i;
    str_value_ = oskar_settings_utility_int_to_string(value_);
    return true;
}

bool Int::is_default() const
{
    return value_ == default_;
}

int Int::value() const
{
    return value_;
}

int Int::default_value() const
{
    return default_;
}

bool Int::operator==(const Int& other) const
{
    return value_ == other.value_;
}

bool Int::operator>(const Int& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
