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

#ifndef OSKAR_SETTINGS_TYPE_DOUBLELIST_HPP_
#define OSKAR_SETTINGS_TYPE_DOUBLELIST_HPP_

#include <vector>

#include "oskar_AbstractSettingsType.hpp"

/**
 * @file DoubleList.hpp
 */

namespace oskar {

/**
 * @class DoubleList
 *
 * @brief
 * A list of doubles
 *
 * @details
 * Set with a CSV list.
 * Returns a CSV string list.
 */

class DoubleList : public AbstractSettingsType
{
public:
    DoubleList();
    virtual ~DoubleList();

    bool init(const std::string& s);
    bool set_default(const std::string& s);
    std::string get_default() const;
    bool set_value(const std::string& s);
    std::string get_value() const;
    bool is_default() const;

    std::vector<double> values() const;

    bool operator==(const DoubleList& other) const;
    bool operator>(const DoubleList& other) const;

private:
    bool from_string_(std::vector<double>& values, const std::string& s) const;
    std::string to_string_(const std::vector<double>& values) const;
    std::vector<double> value_;
    std::vector<double> default_;
    char delimiter_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DOUBLELIST_HPP_ */
