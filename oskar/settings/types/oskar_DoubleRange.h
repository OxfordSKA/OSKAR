/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_TYPE_DOUBLERANGE_H_
#define OSKAR_SETTINGS_TYPE_DOUBLERANGE_H_

/**
 * @file oskar_DoubleRange.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"

namespace oskar {

/**
 * @class DoubleRange
 *
 * @brief
 * Ranged double value.
 *
 * @details
 * Initialised with a two entry CSV list consisting of the minimum and
 * maximum range. The range of the allowed value is inclusive.
 *
 * e.g. a range of 3.0,10.0 allows any double x, in the range 3.0 >= x >= 10.0.
 *
 * Values outside the range are set to the closest extreme of the range.
 *
 * By default the range is initialised to a full range of allowed double
 * values (i.e. from -DBL_MAX to DBL_MAX), with a value of 0.0
 */
class DoubleRange : public AbstractSettingsType
{
public:
    enum Format { AUTO, EXPONENT };

    OSKAR_SETTINGS_EXPORT DoubleRange();
    OSKAR_SETTINGS_EXPORT virtual ~DoubleRange();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* s);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT double min() const;
    OSKAR_SETTINGS_EXPORT double max() const;
    OSKAR_SETTINGS_EXPORT double value() const;
    OSKAR_SETTINGS_EXPORT double default_value() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const DoubleRange& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const DoubleRange& other) const;

private:
    bool from_string(double& value, const std::string& s) const;
    Format format_;
    double min_, max_, value_, default_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGE_H_ */
