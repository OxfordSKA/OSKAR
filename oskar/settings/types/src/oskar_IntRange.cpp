/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_IntRange.h"

#include <climits>
#include <vector>

using namespace std;

namespace oskar {

IntRange::IntRange()
{
    (void) IntRange::init("");
}

// LCOV_EXCL_START
IntRange::~IntRange()
{
}
// LCOV_EXCL_STOP

bool IntRange::init(const char* s)
{
    min_ = -INT_MAX;
    max_ =  INT_MAX;
    default_ = 0;
    value_ = 0;
    str_default_ = "0";
    str_value_ = "0";

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -INT_MAX to INT_MAX)
    //  - With 1 entry the range is (p[0] to INT_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    //
    // Notes: if p[0] is the string 'MIN' or p[1] is the string 'MAX'
    // these will resolve as -INT_MAX and INT_MAX respectively.
    //
    bool ok = true;
    vector<string> p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u)
    {
        return true;
    }
    else if (p.size() == 1u)
    {
        if (p[0] == "MIN")
        {
            min_ = -INT_MAX;
        }
        else
        {
            min_ = oskar_settings_utility_string_to_int(p[0], &ok);
        }
        return true;
    }
    else if (p.size() == 2u)
    {
        if (p[0] == "MIN")
        {
            min_ = -INT_MAX;
        }
        else
        {
            min_ = oskar_settings_utility_string_to_int(p[0], &ok);
        }
        if (!ok) return false;

        if (p[1] == "MAX")
        {
            max_ = INT_MAX;
        }
        else
        {
            max_ = oskar_settings_utility_string_to_int(p[1], &ok);
        }
        return true;
    }

    // If more than 3 parameters, set the status to false.
    return false;
}

bool IntRange::set_default(const char* value)
{
    bool ok = this->from_string(value, default_);
    str_default_ = oskar_settings_utility_int_to_string(default_);
    if (ok) this->set_value(value);
    return ok;
} // LCOV_EXCL_LINE

bool IntRange::set_value(const char* value)
{
    bool ok = this->from_string(value, value_);
    str_value_ = oskar_settings_utility_int_to_string(value_);
    return ok;
} // LCOV_EXCL_LINE

bool IntRange::is_default() const
{
    return (value_ == default_);
}

int IntRange::value() const
{
    return value_;
}

int IntRange::default_value() const
{
    return default_;
}

int IntRange::min() const
{
    return min_;
}

int IntRange::max() const
{
    return max_;
}

bool IntRange::operator==(const IntRange& other) const
{
    return value_ == other.value_;
}

bool IntRange::operator>(const IntRange& other) const
{
    return value_ > other.value_;
}

bool IntRange::from_string(const string& s, int& value) const
{
    bool ok = true;
    int i = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;

    if (i >= min_ && i <= max_)
    {
        value = i;
        return true;
    }
    else if (i < min_)
    {
        value = min_;
    }
    else if (i > max_)
    {
        value = max_;
    }
    return false;
}

} // namespace oskar
