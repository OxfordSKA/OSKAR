/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_UnsignedInt.h"

using namespace std;

namespace oskar {

UnsignedInt::UnsignedInt()
{
    (void) UnsignedInt::init(0);
}

// LCOV_EXCL_START
UnsignedInt::~UnsignedInt()
{
}
// LCOV_EXCL_STOP

bool UnsignedInt::init(const char* /*s*/)
{
    default_ = 0;
    value_ = 0;
    str_default_ = "0";
    str_value_ = "0";
    return true;
}

bool UnsignedInt::set_default(const char* value)
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(value, &ok);
    if (!ok || i < 0) return false;
    default_ = i;
    str_default_ = oskar_settings_utility_int_to_string(default_);
    this->set_value(value);
    return true;
}

bool UnsignedInt::set_value(const char* value)
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(value, &ok);
    if (!ok || i < 0) return false;
    value_ = i;
    str_value_ = oskar_settings_utility_int_to_string(value_);
    return true;
}

bool UnsignedInt::is_default() const
{
    return value_ == default_;
}

unsigned int UnsignedInt::value() const
{
    return value_;
}

unsigned int UnsignedInt::default_value() const
{
    return default_;
}

bool UnsignedInt::operator==(const UnsignedInt& other) const
{
    return value_ == other.value_;
}

bool UnsignedInt::operator>(const UnsignedInt& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
