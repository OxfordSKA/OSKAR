/*
 * Copyright (c) 2014-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_KEY_HPP_
#define OSKAR_SETTINGS_KEY_HPP_

/**
 * @file oskar_SettingsKey.h
 */

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

#include <vector>
#include <string>

namespace oskar {

/*!
 * @class SettingsKey
 *
 * @brief Class to store and provide utility for settings keys.
 *
 * @details
 * Settings keys in general are a series of delimited keywords.
 */

class OSKAR_SETTINGS_EXPORT SettingsKey
{
public:
    /*! Default constructor */
    SettingsKey(char separator = '/');

    /*! Constructor */
    SettingsKey(const char* key, char separator = '/');

    /*! Destructor */
    ~SettingsKey();

    /*! Return the last element of the key. */
    const char* back() const;

    /*! Return the depth of the key (depth = size - 1). */
    int depth() const;

    /*! Returns true if the key is empty */
    bool empty() const;

    /*! Creates the key from a string. */
    void from_string(const char* key, char separator = '/');

    /*! Return the key separator */
    char separator() const;

    /*! Set the key separator */
    void set_separator(char s = '/');

    /*! Equality operator */
    bool operator==(const SettingsKey& other) const;

    /*! Operator to access sub-elements of the key. */
    const char* operator[](int i) const;

    /*! Implicit conversion operator. */
    operator const char*() const;

private:
    SettingsKey(const SettingsKey&);

    char sep_;
    std::string key_;
    std::vector<std::string> tokens_;
};

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_KEY_HPP_ */
