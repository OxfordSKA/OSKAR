/*
 * Copyright (c) 2015-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_
#define OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_

/**
 * @file oskar_IntRangeExt.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"
#include "settings/extern/ttl/var/variant.hpp"

namespace oskar {

class IntRangeExt : public AbstractSettingsType
{
public:
    typedef ttl::var::variant<int, std::string> Value;

    OSKAR_SETTINGS_EXPORT IntRangeExt();
    OSKAR_SETTINGS_EXPORT virtual ~IntRangeExt();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT int value() const;
    OSKAR_SETTINGS_EXPORT int min() const;
    OSKAR_SETTINGS_EXPORT int max() const;
    OSKAR_SETTINGS_EXPORT const char* ext_min() const;
    OSKAR_SETTINGS_EXPORT const char* ext_max() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const IntRangeExt& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const IntRangeExt& other) const;

private:
    bool from_string(Value& value, const char* s) const;
    std::string to_string(const Value& value) const;

    int min_, max_;
    std::string ext_min_, ext_max_;
    Value default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_ */
