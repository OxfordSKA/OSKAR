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

#ifndef OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_HPP_
#define OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_HPP_

#include <oskar_AbstractSettingsType.hpp>
#include <string>
#include <ttl/var/variant.hpp>

/**
 * @file DoubleRangeExt.hpp
 */

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

    DoubleRangeExt();
    virtual ~DoubleRangeExt();

    bool init(const std::string& s);
    bool set_default(const std::string& value);
    std::string get_default() const;
    bool set_value(const std::string& value);
    std::string get_value() const;
    bool is_default() const;

    bool set_value(double d);
    bool set_default(double d);

    // FIXME(BM) handle string types.
    double value() const { return ttl::var::get<double>(value_); }
    // FIXME(BM) handle string types.
    double default_value() const { return ttl::var::get<double>(default_); }
    double min() const { return min_; }
    double max() const { return max_; }
    std::string ext_min() const { return ext_min_; }
    std::string ext_max() const { return ext_max_; }
    bool is_max() const {
        return (value_.which() == STRING &&
                        ttl::var::get<std::string>(value_) == ext_max_);
    }
    bool is_min() const {
        return (value_.which() == STRING &&
                        ttl::var::get<std::string>(value_) == ext_min_);
    }

    bool operator==(const DoubleRangeExt& other) const;
    bool operator>(const DoubleRangeExt& other) const;

private:
    bool from_double_(Value& value, double d) const;
    bool from_string_(Value& value, const std::string& s) const;
    std::string to_string_(const Value& value) const;

    double min_, max_;
    std::string ext_min_, ext_max_;
    enum value_types { DOUBLE, STRING };
    Format format_;
    Value value_;
    Value default_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGEEXT_HPP_ */
