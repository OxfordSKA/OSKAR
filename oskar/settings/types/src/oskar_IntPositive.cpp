/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_IntPositive.h"

namespace oskar {

static bool from_string(const char* s, int& value)
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;
    if (i >= 1) {
        value = i;
        return true;
    }
    return false;
}

IntPositive::IntPositive() : Int()
{
    (void) IntPositive::init(0);
}

// LCOV_EXCL_START
IntPositive::~IntPositive()
{
}
// LCOV_EXCL_STOP

bool IntPositive::init(const char* /*s*/)
{
    default_ = 1;
    value_ = 1;
    str_default_ = "1";
    str_value_ = "1";
    return true;
}

bool IntPositive::set_default(const char* value)
{
    bool ok = from_string(value, default_);
    str_default_ = oskar_settings_utility_int_to_string(default_);
    if (ok) this->set_value(value);
    return ok;
} // LCOV_EXCL_LINE

bool IntPositive::set_value(const char* value)
{
    bool ok = from_string(value, value_);
    str_value_ = oskar_settings_utility_int_to_string(value_);
    return ok;
} // LCOV_EXCL_LINE

} // namespace oskar
