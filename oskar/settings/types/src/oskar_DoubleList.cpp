/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_DoubleList.h"
#include <sstream>
#include <cmath>
#include <iostream>

using namespace std;

namespace oskar {

static bool from_string(vector<double>& values, const string& s, char delimiter)
{
    // Convert the string to a vector of doubles.
    vector<double> temp;
    istringstream ss(s);
    string token;
    while (getline(ss, token, delimiter))
    {
        bool valid = true;
        double v = oskar_settings_utility_string_to_double(token, &valid);
        if (!valid) return false;
        temp.push_back(v);
    }
    values = temp;
    return true;
}

static string to_string(const vector<double>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        ss << oskar_settings_utility_double_to_string_2(values[i], 'g');
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

DoubleList::DoubleList()
{
    (void) DoubleList::init(0);
}

// LCOV_EXCL_START
DoubleList::~DoubleList()
{
}
// LCOV_EXCL_STOP

bool DoubleList::init(const char* /*s*/)
{
    // TODO(BM) Set the delimiter from an initialisation string.
    delimiter_ = ',';
    return true;
}

bool DoubleList::set_default(const char* s)
{
    bool ok = from_string(default_, s, delimiter_);
    str_default_ = to_string(default_, delimiter_);
    if (ok) set_value(s);
    return ok;
} // LCOV_EXCL_LINE

bool DoubleList::set_value(const char* s)
{
    bool ok = from_string(value_, s, delimiter_);
    str_value_ = to_string(value_, delimiter_);
    return ok;
} // LCOV_EXCL_LINE

bool DoubleList::is_default() const
{
    return compare_vectors(value_, default_);
}

int DoubleList::size() const
{
    return (int) value_.size();
}

const double* DoubleList::values() const
{
    return value_.size() > 0 ? &value_[0] : 0;
}

bool DoubleList::operator==(const DoubleList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool DoubleList::operator>(const DoubleList&) const
{
    return false;
}

} // namespace oskar
