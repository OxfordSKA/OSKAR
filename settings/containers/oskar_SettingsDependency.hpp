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

#ifndef OSKAR_SETTINGS_DEPENDENCY_HPP_
#define OSKAR_SETTINGS_DEPENDENCY_HPP_

/**
 * @file oskar_SettingsDependency.hpp
 */

#ifdef __cplusplus

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
    SettingsDependency(const std::string& key,
                       const std::string& value,
                       const std::string& logic = "EQ");

    /*! Return the dependency key */
    const std::string& key() const { return key_; }

    /*! Return the dependency value */
    const std::string& value() const { return value_; }

    /*! Return the dependency value logic */
    SettingsDependency::Logic logic() const { return logic_; }

    /*! Return the dependency value logic string */
    const char* logic_string() const;

    /*! Returns true if the dependency has valid key, logic and value. */
    bool is_valid() const;

    /*! Static method to convert logic enum to a string. */
    static const char* logic_to_string(const SettingsDependency::Logic&);

    /*! Static method to convert logic string to logic enum. */
    static SettingsDependency::Logic string_to_logic(const std::string&);

private:
    std::string key_;
    std::string value_;
    SettingsDependency::Logic logic_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* C interface. */
struct oskar_SettingsDependency;
#ifndef OSKAR_SETTINGS_DEPENDENCY_TYPEDEF_
#define OSKAR_SETTINGS_DEPENDENCY_TYPEDEF_
typedef struct oskar_SettingsDependency oskar_SettingsDependency;
#endif /* OSKAR_SETTINGS_DEPENDENCY_TYPEDEF_ */

const char* oskar_settings_dependency_key(
        const oskar_SettingsDependency* dep);
const char* oskar_settings_dependency_value(
        const oskar_SettingsDependency* dep);
const char* oskar_settings_dependency_logic_string(
        const oskar_SettingsDependency* dep);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_DEPENDENCY_HPP_ */
