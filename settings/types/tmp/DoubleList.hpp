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

#ifndef OSKAR_SETTINGS_TYPE_DOUBLELIST_HPP_
#define OSKAR_SETTINGS_TYPE_DOUBLELIST_HPP_

#include <AbstractType.hpp>
#include <vector>
#include <string>

namespace oskar {

class DoubleList : public AbstractType
{
public:
    DoubleList();
    DoubleList(const char* s, char delimiter = ',');
    ~DoubleList();

public: // Implementation of pure virtual methods on AbstractType
    bool isSet() const;
    std::string toString(bool *ok = 0) const;
    void set(const std::string& s, bool* ok = 0);

public:
    // TODO clean this all up!
    // - method to set delimiter
    // - remove toStr and fromStr methods
    // - remove operators?
    // - be slightly more permissive in string separators allow:
    //      - space/tab - delim - space/tab
    size_t size() const;
    void clear();

    const char* toStr(char delimiter = ',') const;
    void fromStr(const char* s, char delimiter = ',', bool* ok = 0);

    double at(size_t i) const;
    double operator[](size_t i) const;
    double& operator[](size_t i);
    DoubleList& operator<<(double i);

    // Equal if they contain the same values in the same order!
    bool isEqual(const DoubleList& other) const;
    bool operator==(const DoubleList& other) const;
    bool operator!=(const DoubleList& other) const;

private:
    std::vector<double> values_;
};

} // namespace oskar

#endif /* OSKAR_SETTINGS_TYPE_DOUBLE_LIST_HPP_ */
