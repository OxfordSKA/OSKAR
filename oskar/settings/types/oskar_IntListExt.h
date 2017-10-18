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

#ifndef OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_
#define OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_

/**
 * @file oskar_IntListExt.h
 */

#include <vector>
#include "settings/types/oskar_AbstractSettingsType.h"
#include "settings/extern/ttl/var/variant.hpp"

namespace oskar {

/**
 * @class IntListExt
 *
 * @brief
 * A list of integers or a single special string
 */
class IntListExt : public AbstractSettingsType
{
 public:
    typedef ttl::var::variant<std::vector<int>, std::string> Value;

 public:
    OSKAR_SETTINGS_EXPORT IntListExt();
    OSKAR_SETTINGS_EXPORT virtual ~IntListExt();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT const char* special_string() const;
    OSKAR_SETTINGS_EXPORT bool is_extended() const;
    OSKAR_SETTINGS_EXPORT int size() const;
    OSKAR_SETTINGS_EXPORT const int* values() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const IntListExt& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const IntListExt&) const;

 private:
    bool from_string_(const std::string& s, Value& val) const;
    std::string to_string_(const Value& v);
    std::string special_value_;
    Value value_, default_;
    char delimiter_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_INTLIST_EXT_H_ */
