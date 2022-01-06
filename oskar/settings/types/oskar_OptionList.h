/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_OPTIONLIST_H_
#define OSKAR_SETTINGS_TYPE_OPTIONLIST_H_

/**
 * @file oskar_OptionList.h
 */

#include <vector>
#include "settings/types/oskar_AbstractSettingsType.h"

namespace oskar {

/**
 * @class OptionList
 *
 * @brief
 * A list of strings with one selectable value.
 *
 * @details
 * Initialised with a CSV list of strings which are to be the set of allowed
 * options.
 */
class OptionList : public AbstractSettingsType
{
public:
    OSKAR_SETTINGS_EXPORT OptionList();
    OSKAR_SETTINGS_EXPORT  virtual ~OptionList();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* s);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT int size() const;
    OSKAR_SETTINGS_EXPORT const char* option(int i) const;

    OSKAR_SETTINGS_EXPORT bool operator==(const OptionList& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const OptionList&) const;

private:
    bool from_string(std::string& value, const std::string& s) const;
    std::vector<std::string> options_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_OPTIONLIST_H_ */
