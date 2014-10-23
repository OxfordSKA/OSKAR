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

#ifndef OSKAR_SETTINGS_TYPE_DATETIME_HPP_
#define OSKAR_SETTINGS_TYPE_DATETIME_HPP_

#include <AbstractType.hpp>
#include <string>

namespace oskar {

class DateTime : public AbstractType
{
public:
    DateTime();
    DateTime(const std::string& s);
    virtual ~DateTime();

public: // Implementation of pure virtual methods on AbstractType
    bool isSet() const;
    std::string toString(bool* ok = 0) const;
    void set(const std::string& s, bool *ok = 0);
    bool operator==(const DateTime& other) const;

public:
    void clear();
    int year() const;
    int month() const;
    int day() const;
    int hours() const;
    int minutes() const;
    double seconds() const;
    void set(int year, int month, int day, int hour = 0, int minutes = 0,
            double seconds = 0.0);
    bool isEqual(const DateTime& other) const;
    std::string formatString() const;

private:
    int year_;
    int month_;
    int day_;
    int hour_;
    int minute_;
    double seconds_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_DATETIME_HPP_ */
