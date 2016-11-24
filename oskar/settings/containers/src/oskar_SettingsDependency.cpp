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

#include "oskar_SettingsKey.hpp"
#include "oskar_SettingsDependency.hpp"

namespace oskar {


SettingsDependency::SettingsDependency(const std::string& key,
                       const std::string& value,
                       const std::string& logic)
: key_(key), value_(value)
{
    logic_ = SettingsDependency::string_to_logic(logic);
}

bool SettingsDependency::is_valid() const
{
    if (logic_ == UNDEF || key_.empty() || value_.empty())
        return false;
    return true;
}

const char* SettingsDependency::logic_string() const
{
    return SettingsDependency::logic_to_string(logic_);
}

const char* SettingsDependency::logic_to_string(
        const SettingsDependency::Logic& v)
{
    switch (v) {
        case EQ: return "EQ";
        case NE: return "NE";
        case GT: return "GT";
        case GE: return "GE";
        case LT: return "LT";
        case LE: return "LE";
        default: return "";
    };
    return "";
}

SettingsDependency::Logic SettingsDependency::string_to_logic(const std::string& s)
{
    Logic l = UNDEF;
    if (s.empty())      l = EQ;
    else if (s == "EQ") l = EQ;
    else if (s == "NE") l = NE;
    else if (s == "GT") l = GT;
    else if (s == "GE") l = GE;
    else if (s == "LT") l = LT;
    else if (s == "LE") l = LE;
    return l;
}

} // namespace oskar

/* C interface. */
struct oskar_SettingsDependency : public oskar::SettingsDependency
{
};

const char* oskar_settings_dependency_key(
        const oskar_SettingsDependency* dep)
{
    return dep->key().c_str();
}

const char* oskar_settings_dependency_value(
        const oskar_SettingsDependency* dep)
{
    return dep->value().c_str();
}

const char* oskar_settings_dependency_logic_string(
        const oskar_SettingsDependency* dep)
{
    return dep->logic_string();
}

