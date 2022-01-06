/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_StringList.h"

#include <sstream>

using namespace std;

namespace oskar {

static string to_string(const vector<string>& values, char delimiter)
{
    ostringstream ss;
    for (size_t i = 0u; i < values.size(); ++i)
    {
        ss << values.at(i);
        if (i < values.size() - 1) ss << delimiter;
    }
    return ss.str();
}

StringList::StringList() : delimiter_(',')
{
}

// LCOV_EXCL_START
StringList::~StringList()
{
}
// LCOV_EXCL_STOP

bool StringList::init(const char* /*s*/)
{
     // TODO(BM) allow a a different delimiter via the init method?
     value_.clear();
     default_.clear();
     pointers_.clear();
     return true;
}

bool StringList::set_default(const char* s)
{
    default_ = oskar_settings_utility_string_get_type_params(s);
    str_default_ = to_string(default_, delimiter_);
    this->set_value(s);
    return true;
} // LCOV_EXCL_LINE

bool StringList::set_value(const char* s)
{
    value_ = oskar_settings_utility_string_get_type_params(s);
    str_value_ = to_string(value_, delimiter_);
    pointers_.clear();
    for (size_t i = 0; i < value_.size(); ++i)
    {
        pointers_.push_back(value_[i].c_str());
    }
    return true;
} // LCOV_EXCL_LINE

bool StringList::is_default() const
{
    return compare_vectors(value_, default_);
}

int StringList::size() const { return (int) value_.size(); }

const char* const* StringList::values() const
{
    return pointers_.size() > 0 ? &pointers_[0] : 0;
}

bool StringList::operator==(const StringList& other) const
{
    return compare_vectors(value_, other.value_);
}

bool StringList::operator>(const StringList&) const
{
    return false;
}

} // namespace oskar
