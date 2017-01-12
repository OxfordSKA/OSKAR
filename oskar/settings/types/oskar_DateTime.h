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

#ifndef OSKAR_SETTINGS_TYPE_DATETIME_H_
#define OSKAR_SETTINGS_TYPE_DATETIME_H_

/**
 * @file oskar_DateTime.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"

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

class OSKAR_SETTINGS_EXPORT DateTime : public AbstractSettingsType
{
public:
    enum Format { UNDEF = -1, BRITISH, CASA, INTERNATIONAL, ISO, MJD };
    class Value {
     public:
        Value()
     : style(DateTime::UNDEF), year(0), month(0), day(0),
       hours(0), minutes(0), seconds(0.0) {}
        void clear() {
            style = DateTime::UNDEF;
            year = 0;
            month = 0;
            day = 0;
            hours = 0;
            minutes = 0;
            seconds = 0.0;
        }
        DateTime::Format style;
        int year, month, day, hours, minutes;
        double seconds;
    };

 public:
    DateTime();
    virtual ~DateTime();

    bool init(const std::string& s);
    bool set_default(const std::string& value);
    std::string get_default() const;
    bool set_value(const std::string& value);
    std::string get_value() const;
    bool is_default() const;

    Value value() const { return value_; }
    Value default_value() const { return default_; }

    double to_mjd() const;
    double to_mjd_2() const;
    void from_mjd(double mjd);
    DateTime::Format format() const { return value_.style; }
    void set_format(DateTime::Format format) {
        value_.style = format;
        default_.style = format;
    }

    bool operator==(const DateTime& other) const;
    bool operator>(const DateTime& other) const;

 private:
    void from_mjd_(double mjd, Value& dateTime) const;
    Value string_to_date_time_(const std::string& s, bool& ok) const;
    std::string date_time_to_string(const Value& dateTime) const;
    bool parse_date_style_1_(const std::string& s, Value& dateTime) const;
    bool parse_date_style_2_(const std::string& s, Value& dateTime) const;
    bool parse_date_style_3_(const std::string& s, Value& dateTime) const;
    bool parse_date_style_4_(const std::string& s, Value& dateTime) const;
    bool parse_time_(const std::string& s, Value& dateTime) const;

 private:
    Value value_;
    Value default_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_DATETIME_H_ */
