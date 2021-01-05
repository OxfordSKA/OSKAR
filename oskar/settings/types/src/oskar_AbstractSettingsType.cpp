/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_AbstractSettingsType.h"

using namespace std;

namespace oskar {

AbstractSettingsType::AbstractSettingsType()
{
}

// LCOV_EXCL_START
AbstractSettingsType::~AbstractSettingsType()
{
}
// LCOV_EXCL_STOP

const char* AbstractSettingsType::get_default() const
{
    return str_default_.c_str();
}

const char* AbstractSettingsType::get_value() const
{
    return str_value_.c_str();
}

} // namespace oskar
