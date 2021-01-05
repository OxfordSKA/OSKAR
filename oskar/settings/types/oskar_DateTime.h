/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
 */
class DateTime : public AbstractSettingsType
{
public:
    enum Format { UNDEF = -1, BRITISH, CASA, INTERNATIONAL, ISO, MJD };
    struct OSKAR_SETTINGS_EXPORT Value
    {
        DateTime::Format style;
        int year, month, day, hours, minutes;
        double seconds;

        Value() : style(DateTime::UNDEF), year(0), month(0), day(0),
                hours(0), minutes(0), seconds(0.0) {}
        void clear()
        {
            style = DateTime::UNDEF;
            year = 0;
            month = 0;
            day = 0;
            hours = 0;
            minutes = 0;
            seconds = 0.0;
        }
    };

 public:
    OSKAR_SETTINGS_EXPORT DateTime();
    OSKAR_SETTINGS_EXPORT virtual ~DateTime();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT Value value() const;
    OSKAR_SETTINGS_EXPORT Value default_value() const;

    OSKAR_SETTINGS_EXPORT double to_mjd() const;
    OSKAR_SETTINGS_EXPORT double to_mjd_2() const;
    OSKAR_SETTINGS_EXPORT void from_mjd(double mjd);
    OSKAR_SETTINGS_EXPORT DateTime::Format format() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const DateTime& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const DateTime& other) const;

 private:
    Value default_, value_;
};

} /* namespace oskar */

#endif /* include guard */
