/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_
#define OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_

/**
 * @file oskar_IntListExt.h
 */

#include <vector>
#include "settings/types/oskar_AbstractSettingsType.h"
#include "settings/extern/ttl/var/variant.hpp"

namespace oskar {

/**
 * @class IntListExt
 *
 * @brief
 * A list of integers or a single special string
 */
class IntListExt : public AbstractSettingsType
{
 public:
    typedef ttl::var::variant<std::vector<int>, std::string> Value;

 public:
    OSKAR_SETTINGS_EXPORT IntListExt();
    OSKAR_SETTINGS_EXPORT virtual ~IntListExt();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT const char* special_string() const;
    OSKAR_SETTINGS_EXPORT bool is_extended() const;
    OSKAR_SETTINGS_EXPORT int size() const;
    OSKAR_SETTINGS_EXPORT const int* values() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const IntListExt& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const IntListExt&) const;

 private:
    bool from_string(const std::string& s, Value& val) const;
    std::string to_string(const Value& v) const;
    std::string special_value_;
    Value value_, default_;
    char delimiter_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_ */
