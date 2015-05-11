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

#ifndef OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_
#define OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_

#include "oskar_AbstractType.hpp"


/**
 * @file DoubleRange.hpp
 */

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

class DoubleRange : public AbstractType
{
public:
    DoubleRange();
    virtual ~DoubleRange();
    void init(const std::string& s, bool* ok = 0);
    void fromString(const std::string& s, bool* ok = 0);
    std::string toString() const;
    std::string toString(const std::string& fmt) const;

    double range_min() const { return min_; }
    double range_max() const { return max_; }

private:
    double min_, max_, value_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DOUBLERANGE_HPP_ */
