/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_Double.h"
#include "settings/oskar_settings_utility_string.h"
#include <cfloat>
#include <cmath>

namespace oskar {

Double::Double()
{
    (void) Double::init(0);
}

// LCOV_EXCL_START
Double::~Double()
{
}
// LCOV_EXCL_STOP

bool Double::init(const char* /*s*/)
{
    format_ = AUTO;
    default_ = 0.0;
    value_ = 0.0;
    str_default_ = "0.0";
    str_value_ = "0.0";
    return true;
}

bool Double::set_default(const char* value)
{
    bool ok = true;
    std::string val(value);
    double d = oskar_settings_utility_string_to_double(val, &ok);
    if (!ok) return false;
    format_ = (val.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    default_ = d;
    str_default_ = oskar_settings_utility_double_to_string_2(default_,
            (format_ == AUTO ? 'g' : 'e'));
    this->set_value(value);
    return true;
}

bool Double::set_value(const char* value)
{
    bool ok = true;
    std::string val(value);
    double d = oskar_settings_utility_string_to_double(val, &ok);
    if (!ok) return false;
    format_ = (val.find_first_of('e') != std::string::npos) ? EXPONENT : AUTO;
    value_ = d;
    str_value_ = oskar_settings_utility_double_to_string_2(value_,
            (format_ == AUTO ? 'g' : 'e'));
    return true;
}

bool Double::is_default() const
{
    return (fabs(value_ - default_) < DBL_EPSILON);
}

double Double::value() const
{
    return value_;
}

double Double::default_value() const
{
    return default_;
}

bool Double::operator==(const Double& other) const
{
    return (fabs(value_ - other.value_) < DBL_EPSILON);
}

bool Double::operator>(const Double& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
