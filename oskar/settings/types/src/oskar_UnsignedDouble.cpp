/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_UnsignedDouble.h"
#include "settings/oskar_settings_utility_string.h"

using namespace std;

namespace oskar {

UnsignedDouble::UnsignedDouble() : Double()
{
}

// LCOV_EXCL_START
UnsignedDouble::~UnsignedDouble()
{
}
// LCOV_EXCL_STOP

bool UnsignedDouble::set_default(const char* value)
{
    double v = 0.0;
    bool ok = true;
    string val(value);
    format_ = (val.find_first_of('e') != string::npos) ? EXPONENT : AUTO;
    v = oskar_settings_utility_string_to_double(val, &ok);
    if (v < 0.0) return false;
    default_ = v;
    str_default_ = oskar_settings_utility_double_to_string_2(default_,
            (format_ == AUTO ? 'g' : 'e'));
    if (ok) set_value(value); else (void) init(0);
    return ok;
}

bool UnsignedDouble::set_value(const char* value)
{
    double v = 0.0;
    bool ok = true;
    string val(value);
    format_ = (val.find_first_of('e') == string::npos) ? AUTO : EXPONENT;
    v = oskar_settings_utility_string_to_double(val, &ok);
    if (v < 0.0) return false;
    value_ = v;
    str_value_ = oskar_settings_utility_double_to_string_2(value_,
            (format_ == AUTO ? 'g' : 'e'));
    return ok;
}

} // namespace oskar
