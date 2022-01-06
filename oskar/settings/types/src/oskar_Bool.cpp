/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_Bool.h"

namespace oskar {

Bool::Bool()
{
    (void) Bool::init(0);
}

// LCOV_EXCL_START
Bool::~Bool()
{
}
// LCOV_EXCL_STOP

bool Bool::init(const char* /*s*/)
{
    default_ = false;
    value_ = false;
    str_default_ = "false";
    str_value_ = "false";
    return true;
}

bool Bool::set_default(const char* s)
{
    default_ = oskar_settings_utility_string_starts_with(s, "TRUE", false) ||
            oskar_settings_utility_string_starts_with(s, "ON", false) ||
            oskar_settings_utility_string_starts_with(s, "1", false);
    str_default_ = default_ ? "true" : "false";
    this->set_value(s);
    return true;
} // LCOV_EXCL_LINE

bool Bool::set_value(const char* s)
{
    value_ = oskar_settings_utility_string_starts_with(s, "TRUE", false) ||
            oskar_settings_utility_string_starts_with(s, "ON", false) ||
            oskar_settings_utility_string_starts_with(s, "1", false);
    str_value_ = value_ ? "true" : "false";
    return true;
} // LCOV_EXCL_LINE

bool Bool::is_default() const
{
    return value_ == default_;
}

bool Bool::value() const
{
    return value_;
}

bool Bool::operator==(const Bool& other) const
{
    return value_ == other.value_;
}

bool Bool::operator>(const Bool&) const
{
    return false;
}

} // namespace oskar
