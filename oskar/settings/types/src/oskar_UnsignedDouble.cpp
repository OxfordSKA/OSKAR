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

#include "settings/types/oskar_UnsignedDouble.h"
#include "settings/oskar_settings_utility_string.h"

using namespace std;

namespace oskar {

UnsignedDouble::UnsignedDouble() : Double()
{
}

UnsignedDouble::~UnsignedDouble()
{
}

bool UnsignedDouble::set_default(const char* value)
{
    double v = 0.0;
    bool ok = true;
    string val(value);
    format_ = (val.find_first_of('e') != string::npos) ? EXPONENT : AUTO;
    v = oskar_settings_utility_string_to_double(val, &ok);
    if (v < 0.0) return false;
    default_ = v;
    str_default_ = oskar_settings_utility_double_to_string_2(default_,
            (format_ == AUTO ? 'g' : 'e'));
    if (ok)
        set_value(value);
    else
        (void) init(0);
    return ok;
}

bool UnsignedDouble::set_value(const char* value)
{
    double v = 0.0;
    bool ok = true;
    string val(value);
    format_ = (val.find_first_of('e') == string::npos) ? AUTO : EXPONENT;
    v = oskar_settings_utility_string_to_double(val, &ok);
    if (v < 0.0) return false;
    value_ = v;
    str_value_ = oskar_settings_utility_double_to_string_2(value_,
            (format_ == AUTO ? 'g' : 'e'));
    return ok;
}

} // namespace oskar
