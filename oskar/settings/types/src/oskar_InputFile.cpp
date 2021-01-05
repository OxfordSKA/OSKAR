/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_InputFile.h"
#include "settings/oskar_settings_utility_string.h"

namespace oskar {

InputFile::InputFile() : String()
{
}

// LCOV_EXCL_START
InputFile::~InputFile()
{
}
// LCOV_EXCL_STOP

bool InputFile::set_default(const char* value)
{
    str_default_ = oskar_settings_utility_string_trim(value);
    str_value_ = str_default_;
    return true;
} // LCOV_EXCL_LINE

bool InputFile::set_value(const char* value)
{
    str_value_ = oskar_settings_utility_string_trim(value);
    return true;
} // LCOV_EXCL_LINE

} // namespace oskar
