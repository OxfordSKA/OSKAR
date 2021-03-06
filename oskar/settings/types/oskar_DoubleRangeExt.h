/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
    bool from_string_(Value& value, const std::string& s) const;
    std::string to_string_(const Value& value) const;

    double min_, max_;
    std::string ext_min_, ext_max_;
    Format format_;
    Value default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_H_ */
