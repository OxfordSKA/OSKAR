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

#ifndef OSKAR_SETTINGS_TYPE_DATETIME_HPP_
#define OSKAR_SETTINGS_TYPE_DATETIME_HPP_

#include <vector>
#include <oskar_AbstractType.hpp>

/**
 * @file oskar_DateTime.hpp
 */

namespace oskar {

/**
 * @class DateTime
 *
 * @brief
 *
 * @details
 * TODO check leading zero behaviour in OSKAR ...
 *     oskar_settings_load_observation.cpp
 *
 *  TODO also ready MJD
 *
 *  d-M-yyyy h:m:s[.z] - British style
 *  yyyy/M/d/h:m:s[.z] - CASA style
 *  yyyy-M-d h:m:s[.z] - International style
 *  yyyy-M-dTh:m:s[.z] - ISO date style
 *
 *
 */

class DateTime : public AbstractType
{
public:
    DateTime();
    virtual ~DateTime();
    void init(const std::string& s, bool* ok = 0);
    void fromString(const std::string& s, bool* ok = 0);
    std::string toString() const;

    int year() const;
    int month() const;
    int day() const;
    int hours() const;
    int minutes() const;
    double seconds() const;

    //double mjd() const;

private:
    void parse_date_style_1_(const std::string& s, bool* ok = 0);
    void parse_date_style_2_(const std::string& s, bool* ok = 0);
    void parse_date_style_3_(const std::string& s, bool* ok = 0);
    void parse_date_style_4_(const std::string& s, bool* ok = 0);
    void parse_time_(const std::string& s, bool* ok = 0);

private:
    enum date_format { UNDEF, BRITISH, CASA, INTERNATIONAL, ISO, MJD };
    int style_;
    int year_;
    int month_;
    int day_;
    int hours_;
    int minutes_;
    double seconds_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DATETIME_HPP_ */
