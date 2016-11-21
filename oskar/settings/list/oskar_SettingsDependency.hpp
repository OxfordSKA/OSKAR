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

#ifndef OSKAR_SETTINGS_DEPENDENCY_HPP_
#define OSKAR_SETTINGS_DEPENDENCY_HPP_

/**
 * @file oskar_SettingsDependency.hpp
 */

namespace oskar {

class Dependency
{
public:
    enum Combination { UNDEF, AND, OR };
    enum Logic { EQ, NE, GT, GE, LT, LE };

public:
    // ctor
    Dependency(const SettingsKey& k, const Variant& v,
            const Dependency::Logic& l, const Dependency::Combination& c,
            int depth);

    // Accessors
    SettingsKey key() const;
    Variant value() const;
    Logic logic() const;
    Combination combination() const;
    int depth() const;

    // String conversions
    std::string logicString() const;
    Combination getCombination(const std::string& s) const;

    // Static methods
    static std::string logicString(const Logic& v);
    static Logic logicFromString(const std::string& v);
    static Combination combinationFromString(const std::string& s);

private:
    SettingsKey key_;
    Variant value_;
    Logic logic_;
    Combination comb_;
    int depth_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_DEPENDENCY_HPP_ */
