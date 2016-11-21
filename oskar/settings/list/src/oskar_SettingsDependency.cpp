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


#include <oskar_SettingsDependency.hpp>
#include <oskar_SettingsKey.hpp>
#include "../oskar_SettingsVariant.hpp"

namespace oskar {


Dependency::Dependency(const SettingsKey& k, const Variant& v,
        const Dependency::Logic& l, const Dependency::Combination& c, int depth)
: key_(k), value_(v), logic_(l), comb_(c), depth_(depth)
{

}

SettingsKey Dependency::key() const
{
    return key_;
}


Variant Dependency::value() const
{
    return value_;
}


Dependency::Logic Dependency::logic() const
{
    return logic_;
}

Dependency::Combination Dependency::combination() const
{
    return comb_;
}

int Dependency::depth() const
{
    return depth_;
}

std::string Dependency::logicString() const
{
    return Dependency::logicString(logic_);
}

Dependency::Combination Dependency::getCombination(const std::string& s) const
{
    return Dependency::combinationFromString(s);
}

std::string Dependency::logicString(const Dependency::Logic& v)
{
    switch (v) {
        case EQ: return "==";
        case NE: return "!=";
        case GT: return ">";
        case GE: return ">=";
        case LT: return "<";
        case LE: return "<=";
    };
    return "";
}

Dependency::Logic Dependency::logicFromString(const std::string& s)
{
    Logic l = EQ;
    if (s == "EQ") l = EQ;
    else if (s == "NE") l = NE;
    else if (s == "GT") l = GT;
    else if (s == "GE") l = GE;
    else if (s == "LT") l = LT;
    else if (s == "LE") l = LE;
    return l;
}

Dependency::Combination Dependency::combinationFromString(const std::string& s)
{
    Combination c = UNDEF;
    if (s == "AND") c = AND;
    else if (s == "OR") c = OR;
    return c;
}

} // namespace oskar
