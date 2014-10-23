/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_SETTINGS_TYPE_INTLIST_HPP_
#define OSKAR_SETTINGS_TYPE_INTLIST_HPP_

#include <AbstractType.hpp>
#include <vector>
#include <sstream>
#include <string>

#include <ttl/var/variant.hpp>

namespace VAR = ttl::var;

namespace oskar {

class IntList : public AbstractType
{
public:
    // If the string used to construct the list is not a list of integers
    // it is added to the allowed strings that 'extend' the list.
    // i.e. constructing a list with IntListExt("all") will allow it to hold
    // any integer list or the string all.
    // Additional extension strings can be added, if required, via the
    // add allowed string method.
    IntList(const std::string& s, char delimiter = ',');
    IntList();
    ~IntList();

public:
    bool isSet() const;
    std::string toString(bool* ok = 0) const;
    void set(const std::string& s, bool* ok = 0);

public:
    size_t size() const;
    void clear();
    void fromString(const std::string& s, char delimiter = ',', bool* ok = 0);
    int at(size_t i) const;
    int operator[](size_t i) const;
    void set(size_t index, int i);
    IntList& operator<<(int i);

    bool isValid() const;
    bool isList() const;
    bool isText() const;

    // Equal if they contain the same values in the same order!
    bool isEqual(const IntList& other) const;
    bool operator==(const IntList& other) const;
    bool operator!=(const IntList& other) const;

    void addAllowedString(const std::string& s);
    size_t numAllowedStrings() const;
    std::string allowedString(size_t i) const;

private:
    std::vector<std::string> allowedStrings_;
    char delimiter_;
    VAR::variant<std::vector<int>, std::string> values_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_INTLIST_HPP_ */
