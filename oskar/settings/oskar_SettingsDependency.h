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

#ifndef OSKAR_SETTINGS_DEPENDENCY_HPP_
#define OSKAR_SETTINGS_DEPENDENCY_HPP_

/**
 * @file oskar_SettingsDependency.h
 */

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus
#include <string>

namespace oskar {

/*!
 * @class SettingsDependency
 *
 * @brief Class used to store the dependency between settings.
 *
 * @details
 * Container class used to represent dependencies between settings.
 *
 * A dependency is an association between settings which makes a requirement
 * on the value of another setting for the current setting to be enabled.
 *
 * Dependencies are therefore specified by a key used to retrieve the value
 * of the setting to be check, the required value and value logic.
 */

class SettingsDependency
{
public:
    /*! Value logic enum. */
    enum Logic { UNDEF, EQ, NE, GT, GE, LT, LE };

    /*! Constructor. */
    OSKAR_SETTINGS_EXPORT SettingsDependency(const char* key,
            const char* value,
            const char* logic = 0);

    /*! Return the dependency key */
    OSKAR_SETTINGS_EXPORT const char* key() const;

    /*! Return the dependency value */
    OSKAR_SETTINGS_EXPORT const char* value() const;

    /*! Return the dependency value logic */
    OSKAR_SETTINGS_EXPORT SettingsDependency::Logic logic() const;

    /*! Return the dependency value logic string */
    OSKAR_SETTINGS_EXPORT const char* logic_string() const;

    /*! Returns true if the dependency has valid key, logic and value. */
    OSKAR_SETTINGS_EXPORT bool is_valid() const;

    /*! Static method to convert logic enum to a string. */
    OSKAR_SETTINGS_EXPORT static const char* logic_to_string(
            const SettingsDependency::Logic&);

    /*! Static method to convert logic string to logic enum. */
    OSKAR_SETTINGS_EXPORT static SettingsDependency::Logic string_to_logic(const char*);

private:
    std::string key_, value_;
    SettingsDependency::Logic logic_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_DEPENDENCY_HPP_ */
