/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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
    bool from_string_(double& value, const std::string& s) const;
    Format format_;
    double min_, max_, value_, default_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGE_H_ */
