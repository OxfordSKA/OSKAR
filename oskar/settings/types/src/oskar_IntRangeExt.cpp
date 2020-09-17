/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/types/oskar_IntRangeExt.h"
#include "settings/oskar_settings_utility_string.h"
#include <climits>
#include <vector>
#include <iostream>

using namespace std;
using ttl::var::get;

namespace oskar {

enum value_types { INT, STRING };

static bool compare(const IntRangeExt::Value& a, const IntRangeExt::Value& b)
{
    if (a.is_singular() || b.is_singular()) return false;
    if (a.which() != b.which()) return false;
    if (a.which() == STRING) return (get<string>(a) == get<string>(b));
    if (a.which() == INT) return (get<int>(a) == get<int>(b));
    return false;
}

IntRangeExt::IntRangeExt()
{
    (void) init("");
}

IntRangeExt::~IntRangeExt()
{
}

bool IntRangeExt::init(const char* s)
{
    ext_min_.clear();
    ext_max_.clear();
    min_ = -INT_MAX;
    max_ =  INT_MAX;
    value_ = 0;
    default_ = 0;
    str_default_ = "0";
    str_value_ = "0";

    // Extract range from the parameter CSV string.
    // Parameters, p, for IntRangeExt should be of length between 3 and 4
    //  - With 3 entries the range from (p[0] to p[1] with special string [2]
    //    for values <= p[0]
    //  - With 4 entries the range from (p[0] to p[1] with special string p[2]
    //    for values <= p[0] and special string p[3] for values >= p[1]
    //
    // Note:
    // - p[0] can never be -INT_MAX and for 4 entries p[1] can't be bigger than
    //   INT_MAX-1
    // - For 3 entries, if p[1] is the string 'MAX' the max range is set to
    //   INT_MAX (note this is not possible when there is also a max string)
    //
    bool ok = true;
    vector<string> p = oskar_settings_utility_string_get_type_params(s);
    // For less than 3 values, just use IntRange instead!
    if (p.size() < 3u || p.size() > 4u) {
        return false;
    }
    else if (p[0] == "-INT_MAX" || p[0] == "-MAX" ||
                    p[0] == "INT_MIN" || p[0] == "MIN")
        min_ = -INT_MAX;
    else
        min_ = oskar_settings_utility_string_to_int(p[0], &ok);
    if (!ok) return false;
    if (p[1] == "INT_MAX" || p[1] == "MAX")
        max_ = INT_MAX;
    else
        max_ = oskar_settings_utility_string_to_int(p[1], &ok);
    ext_min_ = p[2];
    if (p.size() == 4u) {
        if (max_ == INT_MAX) return false;
        ext_max_ = p[3];
    }
    return ok;
}

bool IntRangeExt::set_default(const char* value)
{
    bool ok = from_string(default_, value);
    str_default_ = to_string(default_);
    if (ok)
        set_value(value);
    return ok;
}

bool IntRangeExt::set_value(const char* value)
{
    bool ok = from_string(value_, value);
    str_value_ = to_string(value_);
    return ok;
}

bool IntRangeExt::is_default() const
{
    return compare(value_, default_);
}

int IntRangeExt::value() const
{
    if (value_.which() == STRING && get<string>(value_) == ext_min_)
        return min_;
    if (value_.which() == STRING && get<string>(value_) == ext_max_)
        return max_;
    return get<int>(value_);
}

int IntRangeExt::min() const
{
    return min_;
}

int IntRangeExt::max() const
{
    return max_;
}

const char* IntRangeExt::ext_min() const
{
    return ext_min_.c_str();
}

const char* IntRangeExt::ext_max() const
{
    return ext_max_.c_str();
}

bool IntRangeExt::operator==(const IntRangeExt& other) const
{
    return compare(value_, other.value_);
}

bool IntRangeExt::operator>(const IntRangeExt& other) const
{
    if (value_.is_singular() || other.value_.is_singular()) return false;
    if (value_.which() == other.value_.which()) return false;
    if (value_.which() == STRING) return false;
    if (value_.which() == INT) return (get<int>(value_) > get<int>(default_));
    return false;
}

bool IntRangeExt::from_string(Value& value, const char* s) const
{
    if (!ext_min_.empty() && s == ext_min_)
    {
        value = ext_min_;
        return true;
    }
    if (!ext_max_.empty() && s == ext_max_)
    {
        value = ext_max_;
        return true;
    }
    bool ok = true;
    int v = oskar_settings_utility_string_to_int(s, &ok);
    if (!ok) return false;
    if (v == min_ && !ext_min_.empty())
    {
        value = ext_min_;
        return true;
    }
    if (v == max_ && !ext_max_.empty())
    {
        value = ext_max_;
        return true;
    }
    if (v >= min_ && v <= max_)
    {
        value = v;
        return true;
    }
    return false;
}

string IntRangeExt::to_string(const Value& value) const
{
    if (value.is_singular()) return string();
    if (value.which() == INT)
    {
        int v_ = get<int>(value);
        if (v_ == min_ && !ext_min_.empty()) return ext_min_;
        if (v_ == max_ && !ext_max_.empty()) return ext_max_;
        return oskar_settings_utility_int_to_string(v_);
    }
    else if (value.which() == STRING)
        return get<string>(value);
    return string();
}

} // namespace oskar
