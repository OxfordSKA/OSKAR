/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_IntList.h"
#include <sstream>

using namespace std;

namespace oskar {

static bool from_string(const string& s, vector<int>& values, char delimiter)
{
    // Convert the string to a vector of ints.
    vector<int> temp;
    istringstream ss(s);
    string token;
    while (getline(ss, token, delimiter))
    {
        bool valid = true;
        int v = oskar_settings_utility_string_to_int(token, &valid);
        if (!valid) return false;
        temp.push_back(v);
    }
    values = temp;
    return true;
}

static string to_string(const vector<int>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0; i < values.size(); ++i)
    {
        ss << values.at(i);
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

IntList::IntList()
{
    (void) IntList::init(0);
}

// LCOV_EXCL_START
IntList::~IntList()
{
}
// LCOV_EXCL_STOP

bool IntList::init(const char* /*s*/)
{
    // TODO(BM) Could use this to set the delimiter ... ?
    delimiter_ = ',';
    return true;
}

bool IntList::set_default(const char* value)
{
    bool ok = from_string(value, default_, delimiter_);
    str_default_ = to_string(default_, delimiter_);
    if (ok) set_value(value);
    return ok;
} // LCOV_EXCL_LINE

bool IntList::set_value(const char* value)
{
    bool ok = from_string(value, value_, delimiter_);
    str_value_ = to_string(value_, delimiter_);
    return ok;
} // LCOV_EXCL_LINE

bool IntList::is_default() const
{
    return compare_vectors(value_, default_);
}

int IntList::size() const
{
    return (int) value_.size();
}

const int* IntList::values() const
{
    return value_.size() > 0 ? &value_[0] : 0;
}

bool IntList::operator==(const IntList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool IntList::operator>(const IntList&) const
{
    return false;
}

} // namespace oskar
