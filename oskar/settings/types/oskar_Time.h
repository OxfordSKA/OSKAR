/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_TYPE_TIME_H_
#define OSKAR_SETTINGS_TYPE_TIME_H_

/**
 * @file oskar_Time.h
 */

#include <vector>
#include "settings/types/oskar_AbstractSettingsType.h"

namespace oskar {

/**
 * @class Time
 *
 * @brief
 *
 * @details
 * Can be set as a time string or a time in seconds.
 * Accepted time strings:
 *  hh:mm:ss.zzzzzz
 *  s.zzzzzzzz
 */
class Time : public AbstractSettingsType
{
public:
    enum Format { UNDEF = -1, TIME_STRING, SECONDS };
    struct OSKAR_SETTINGS_EXPORT Value
    {
        int hours, minutes;
        double seconds;
        Time::Format format;
        Value() : hours(0), minutes(0), seconds(0.0), format(UNDEF) {}
        void clear() { hours = 0; minutes = 0; seconds = 0.0; format = UNDEF;}
    };

    OSKAR_SETTINGS_EXPORT Time();
    OSKAR_SETTINGS_EXPORT virtual ~Time();

    OSKAR_SETTINGS_EXPORT bool init(const char* param);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT Value value() const;
    OSKAR_SETTINGS_EXPORT double to_seconds() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const Time& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const Time& other) const;

private:
    Value default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_TIME_H_ */
