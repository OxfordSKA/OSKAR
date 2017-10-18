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

#ifndef OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_
#define OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_

/**
 * @file oskar_IntRangeExt.h
 */

#include "settings/types/oskar_AbstractSettingsType.h"
#include "settings/extern/ttl/var/variant.hpp"

namespace oskar {

class IntRangeExt : public AbstractSettingsType
{
public:
    typedef ttl::var::variant<int, std::string> Value;

    OSKAR_SETTINGS_EXPORT IntRangeExt();
    OSKAR_SETTINGS_EXPORT virtual ~IntRangeExt();

    OSKAR_SETTINGS_EXPORT bool init(const char* s);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT bool is_default() const;

    OSKAR_SETTINGS_EXPORT int value() const;
    OSKAR_SETTINGS_EXPORT int min() const;
    OSKAR_SETTINGS_EXPORT int max() const;
    OSKAR_SETTINGS_EXPORT const char* ext_min() const;
    OSKAR_SETTINGS_EXPORT const char* ext_max() const;

    OSKAR_SETTINGS_EXPORT bool operator==(const IntRangeExt& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const IntRangeExt& other) const;

private:
    bool from_string(Value& value, const char* s) const;

    int min_, max_;
    std::string ext_min_, ext_max_;
    Value default_, value_;
};

} /* namespace oskar */

#endif /* OSKAR_SETTINGS_TYPE_INTRANGEEXT_H_ */
