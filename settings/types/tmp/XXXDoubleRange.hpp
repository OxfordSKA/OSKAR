/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_
#define OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_

#include <AbstractType.hpp>

#include <ttl/var/variant.hpp>
#include <string>

namespace oskar {

class DoubleRange : public AbstractType
{
public:
    DoubleRange();
    DoubleRange(double min, double max, double value = 0);
    DoubleRange(double min, double max, const std::string& ext_min);
    DoubleRange(double min, double max, const std::string& ext_min, const std::string& ext_max);
    virtual ~DoubleRange() {}

public:
    virtual bool isSet() const;
    virtual std::string toString(bool* ok = 0) const;
    virtual void set(const std::string& s, bool* ok = 0);

public:
    void set(double d, bool* ok = 0);
    double getDouble(bool* ok = 0) const;
    double min() const { return min_; }
    double max() const { return max_; }

private:
    double min_;
    double max_;
    std::string ext_min_;
    std::string ext_max_;
    enum value_types { DOUBLE, STD_STRING };
    ttl::var::variant<double, std::string> value_;
};

class DoubleRangeExt : public DoubleRange
{
public:
    DoubleRangeExt() : DoubleRange() {}
    DoubleRangeExt(double min, double max, const std::string& ext_min)
    : DoubleRange(min, max, ext_min) {}
    DoubleRangeExt(double min, double max, const std::string& ext_min,
            const std::string& ext_max)
    : DoubleRange(min, max, ext_min, ext_max) {}
};


} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_ */
