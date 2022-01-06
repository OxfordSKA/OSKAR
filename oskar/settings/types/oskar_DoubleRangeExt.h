/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_H_
#define OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_H_

/**
 * @file oskar_DoubleRangeExt.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"
#include "settings/extern/ttl/var/variant.hpp"

namespace oskar {

/**
 * @class DoubleRangeExt
 *
 * @brief
 * Ranged double value.
 *
 * @details
 * Initialised with a CSV list consisting of the minimum and
 * maximum range and the extended string values.
 *
 * The range of the allowed value is inclusive.
 *
 * e.g. a range of 3.0,10.0,min,max allows any double x, in the range
 * 3.0 >= x >= 10.0. For numbers less < 3.0 the string 'min' will be used and
 * for numbers > 10.0 the string 'max' will be used.
 *
 * Values outside the range are set to the extended string value or
 * the closest extreme of the range if the extreme value string for that side
 * of the range is not set.
 *
 * By default the range is initialised to a full range of allowed double
 * values (i.e. from -DBL_MAX to DBL_MAX), with a value of 0.0 and no extended
 * string values.
 */
class DoubleRangeExt : public AbstractSettingsType
{
public:
    typedef ttl::var::variant<double, std::string> Value;
    enum Format { AUTO, EXPONENT };

    OSKAR_SETTINGS_EXPORT DoubleRangeExt();
    OSKAR_SETTINGS_EXPORT virtual ~DoubleRangeExt();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT double value() const;
    OSKAR_SETTINGS_EXPORT double min() const;
    OSKAR_SETTINGS_EXPORT double max() const;
    OSKAR_SETTINGS_EXPORT const char* ext_min() const;
    OSKAR_SETTINGS_EXPORT const char* ext_max() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const DoubleRangeExt& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const DoubleRangeExt& other) const;

private:
    bool from_string(Value& value, const std::string& s) const;
    std::string to_string(const Value& value) const;

    double min_, max_;
    std::string ext_min_, ext_max_;
    Format format_;
    Value default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_H_ */
