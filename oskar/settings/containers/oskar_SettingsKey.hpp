/*
 * Copyright (c) 2014-2015, The University of Oxford
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
 * @file oskar_SettingsKey.hpp
 */

#ifdef __cplusplus

#include <iostream>
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

class SettingsKey
{
public:
    /*! Default constructor */
    SettingsKey(char separator = '/');

    /*! Constructor */
    SettingsKey(const std::string& key, char separator = '/');

    /*! Destructor */
    ~SettingsKey();

    /*! Copy constructor */
    SettingsKey(const SettingsKey&);

    /*! Set the key separator */
    void set_separator(char s = '/');

    /*! Return the key separator */
    char separator() const;

    /*! Returns true if the key is empty */
    bool empty() const;

    /*! Return the depth of the key (depth = size - 1). */
    int depth() const;

    /*! Return the number of separated items in the key */
    int size() const;

    /*! Equality operator */
    bool operator==(const SettingsKey& other) const;

    /*! Implicit conversion operator. */
    operator std::string() const;

    /*! Operator to access sub-elements of the key. */
    std::string operator[](int i) const;

    /*! Return the first element of the key. */
    std::string front() const;

    /*! Return the last element of the key. */
    std::string back() const;

    /*! Return the group key string of the key */
    std::string group() const;

    /*! Convert the key to a const char array */
    const char* c_str() const;

    /*! Stream insertion operator */
    friend std::ostream& operator<<(std::ostream& stream, const SettingsKey& k);

private:
    std::string key_;
    std::vector<std::string> tokens_;
    char sep_;
};

std::ostream& operator<< (std::ostream& stream, const SettingsKey& k);

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_KEY_HPP_ */
