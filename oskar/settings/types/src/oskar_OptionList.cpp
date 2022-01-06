/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_settings_utility_string.h"
#include "settings/types/oskar_OptionList.h"

namespace oskar {

OptionList::OptionList()
{
}

// LCOV_EXCL_START
OptionList::~OptionList()
{
}
// LCOV_EXCL_STOP

bool OptionList::init(const char* s)
{
    options_.clear();
    options_ = oskar_settings_utility_string_get_type_params(s);
    if (options_.size() > 0) set_default(options_[0].c_str());
    return true;
} // LCOV_EXCL_LINE

bool OptionList::set_default(const char* value)
{
    bool ok = this->from_string(str_default_, value);
    if (ok) this->set_value(value);
    return ok;
} // LCOV_EXCL_LINE

bool OptionList::set_value(const char* value)
{
    return this->from_string(str_value_, value);
} // LCOV_EXCL_LINE

bool OptionList::is_default() const { return str_value_ == str_default_; }

bool OptionList::operator==(const OptionList& other) const
{
    return str_value_ == other.str_value_;
}

bool OptionList::operator>(const OptionList&) const
{
    return false;
}

int OptionList::size() const { return (int) options_.size(); }

const char* OptionList::option(int i) const
{
    return i < (int) options_.size() ? options_[i].c_str() : 0;
}

bool OptionList::from_string(std::string& value, const std::string& s) const
{
    for (size_t i = 0; i < options_.size(); ++i)
    {
        if (oskar_settings_utility_string_starts_with(options_[i], s))
        {
            value = options_[i];
            return true;
        }
    }
    return false;
}

} // namespace oskar
