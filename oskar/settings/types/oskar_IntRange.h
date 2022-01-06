/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_INTRANGE_H_
#define OSKAR_SETTINGS_TYPE_INTRANGE_H_

/**
 * @file oskar_IntRange.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"

namespace oskar {

/**
 * @class IntRange
 *
 * @brief
 * Ranged integer value.
 *
 * @details
 * Initialised with a two entry CSV list consisting of the minimum and
 * maximum range. The range of the allowed value is inclusive.
 *
 * e.g. a range of 3,10 allows any integers from 3 to 10, including the
 * values 3 and 10.
 *
 * Values outside the range are set to the closest extreme of the range.
 *
 * By default the range is initialised to a full range of allowed integer
 * values (i.e. from -INT_MAX to INT_MAX), with a value of 0.
 */
class IntRange : public AbstractSettingsType
{
public:
    OSKAR_SETTINGS_EXPORT IntRange();
    OSKAR_SETTINGS_EXPORT virtual ~IntRange();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT int value() const;
    OSKAR_SETTINGS_EXPORT int default_value() const;
    OSKAR_SETTINGS_EXPORT int min() const;
    OSKAR_SETTINGS_EXPORT int max() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const IntRange& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const IntRange& other) const;

private:
    bool from_string(const std::string& s, int& value) const;
    int min_, max_, default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_INTRANGE_H_ */
