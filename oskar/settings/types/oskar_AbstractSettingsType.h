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

#ifndef OSKAR_ABSTRACT_SETTINGS_TYPE_H_
#define OSKAR_ABSTRACT_SETTINGS_TYPE_H_

#include <settings/oskar_settings_macros.h>

/**
 * @file oskar_AbstractSettingsType.h
 */

#include <string>

namespace oskar {

/**
 * @class AbstractSettingsType
 *
 * @brief Interface class for settings types.
 */
class AbstractSettingsType
{
public:
    OSKAR_SETTINGS_EXPORT AbstractSettingsType();
    OSKAR_SETTINGS_EXPORT virtual ~AbstractSettingsType();

    /// Initialises the type from a CSV parameter string.
    OSKAR_SETTINGS_EXPORT virtual bool init(const char* parameters) = 0;

    /// Gets the default as a string.
    OSKAR_SETTINGS_EXPORT const char* get_default() const;

    /// Gets the value as a string.
    OSKAR_SETTINGS_EXPORT const char* get_value() const;

    /// Returns true if the value is the same as the default.
    OSKAR_SETTINGS_EXPORT virtual bool is_default() const = 0;

    /// Sets the default value from a string
    OSKAR_SETTINGS_EXPORT virtual bool set_default(const char* value) = 0;

    /// Sets the value from a string
    OSKAR_SETTINGS_EXPORT virtual bool set_value(const char* value) = 0;

protected:
    std::string str_default_, str_value_;
};

} /* namespace oskar */

#endif /* OSKAR_ABSTRACT_SETTINGS_TYPE_H_ */
