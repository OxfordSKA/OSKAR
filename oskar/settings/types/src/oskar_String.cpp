/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_String.h"

namespace oskar {

String::String()
{
}

// LCOV_EXCL_START
String::~String()
{
}
// LCOV_EXCL_STOP

bool String::init(const char* /*s*/)
{
    str_default_.clear();
    str_value_.clear();
    return true;
}

bool String::set_default(const char* value)
{
    str_default_ = value;
    str_value_ = str_default_;
    return true;
}

bool String::set_value(const char* value)
{
    str_value_ = value;
    return true;
}

bool String::is_default() const
{
    return str_value_ == str_default_;
}

bool String::operator==(const String& other) const
{
    return str_value_ == other.str_value_;
}

bool String::operator>(const String&) const
{
    return false;
}

} // namespace oskar
