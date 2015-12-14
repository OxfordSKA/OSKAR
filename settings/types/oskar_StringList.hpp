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

#ifndef OSKAR_SETTINGS_TYPE_STRINGLIST_HPP_
#define OSKAR_SETTINGS_TYPE_STRINGLIST_HPP_

#include <string>
#include <vector>

#include "oskar_AbstractSettingsType.hpp"

/**
 * @file StringList.hpp
 */

namespace oskar {

/**
 * @class StringList
 *
 * @brief
 * A list of strings.
 *
 * @details
 * Handles CSV lists of strings.
 *
 */

class StringList : public AbstractSettingsType
{
public:
    StringList();
    virtual ~StringList();

    bool init(const std::string& param);
    bool set_default(const std::string& value);
    std::string get_default() const;
    bool set_value(const std::string& value);
    std::string get_value() const;
    bool is_default() const;

    int size() const { return value_.size(); }
    std::vector<std::string> values() const { return value_; }
    std::vector<std::string> default_values() const { return default_; }

    bool operator==(const StringList& other) const;
    bool operator>(const StringList& other) const;

private:
    std::string to_string_(const std::vector<std::string>& values) const;
    std::vector<std::string> default_, value_;
    char delimiter_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_STRINGLIST_HPP_ */
