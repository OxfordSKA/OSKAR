/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_DoubleRange.h"

#include <cfloat>
#include <vector>
#include <cmath>
#include <cfloat>
#include <iostream>

using namespace std;

namespace oskar {

DoubleRange::DoubleRange()
{
    DoubleRange::init("");
}

// LCOV_EXCL_START
DoubleRange::~DoubleRange()
{
}
// LCOV_EXCL_STOP

bool DoubleRange::init(const char* s)
{
    min_   = -DBL_MAX;
    max_   =  DBL_MAX;
    value_ = 0.0;
    default_ = 0.0;
    str_default_ = "0.0";
    str_value_ = "0.0";
    format_ = AUTO;

    // Extract range from the parameter CSV string.
    // Parameters, p, for DoubleRange should be length 0, 1 or 2.
    //  - With 0 entries the range is unchanged (from -DBL_MAX to DBL_MAX)
    //  - With 1 entry the range is (p[0] to DBL_MAX)
    //  - With 2 entries the range is (p[0] to p[1])
    //
    // Notes: if p[0] is the string 'MIN' or p[1] is the string 'MAX'
    // these will resolve as -DBL_MAX and DBL_MAX respectively.
    bool ok = true;
    vector<string> p = oskar_settings_utility_string_get_type_params(s);
    if (p.size() == 0u)
    {
        return false;
    }
    else if (p.size() == 1u)
    {
        if (p[0] == "MIN")
        {
            min_ = -DBL_MAX;
        }
        else
        {
            min_ = oskar_settings_utility_string_to_double(p[0], &ok);
        }
    }
    else if (p.size() == 2u)
    {
        if (p[0] == "MIN")
        {
            min_ = -DBL_MAX;
        }
        else
        {
            min_ = oskar_settings_utility_string_to_double(p[0], &ok);
        }
        if (p[1] == "MAX")
        {
            max_ = DBL_MAX;
        }
        else
        {
            max_ = oskar_settings_utility_string_to_double(p[1], &ok);
        }
    }
    else
    {
        return false;
    }
    return ok;
}

bool DoubleRange::set_default(const char* s)
{
    string v(s);
    format_ = (v.find_first_of('e') != string::npos) ? EXPONENT : AUTO;
    bool ok = this->from_string(default_, v);
    str_default_ = oskar_settings_utility_double_to_string_2(default_,
            (format_ == AUTO ? 'g' : 'e'));
    if (ok) set_value(s);
    return ok;
}

bool DoubleRange::set_value(const char* s)
{
    string v(s);
    format_ = (v.find_first_of('e') != string::npos) ? EXPONENT : AUTO;
    bool ok = this->from_string(value_, v);
    str_value_ = oskar_settings_utility_double_to_string_2(value_,
            (format_ == AUTO ? 'g' : 'e'));
    return ok;
}

bool DoubleRange::is_default() const
{
    return (fabs(default_ - value_) < DBL_EPSILON);
}

double DoubleRange::min() const
{
    return min_;
}

double DoubleRange::max() const
{
    return max_;
}

double DoubleRange::value() const
{
    return value_;
}

double DoubleRange::default_value() const
{
    return default_;
}

bool DoubleRange::from_string(double& value, const string& s) const
{
    bool ok = true;
    double d = oskar_settings_utility_string_to_double(s, &ok);
    if (!ok) return false;
    if (d >= min_ && d <= max_) {
        value = d;
        return true;
    }
    else if (d < min_) {
        value = min_;
        return false;
    }
    else if (d > max_) {
        value = max_;
        return false;
    }
    return false;
}

bool DoubleRange::operator==(const DoubleRange& other) const
{
    return (fabs(value_ - other.value_) < DBL_EPSILON);
}

bool DoubleRange::operator>(const DoubleRange& other) const
{
    return value_ > other.value_;
}

} // namespace oskar
