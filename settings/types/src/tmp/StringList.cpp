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

#include <StringList.hpp>
#include <sstream>

namespace oskar {


StringList::StringList(char delimiter)
: delimiter_(delimiter)
{
}

StringList::StringList(const std::string& s, char delimiter)
: delimiter_(delimiter)
{
    this->set(s);
}

StringList::~StringList()
{
}

bool StringList::isSet() const
{
    if (values_.size() > 0 && !values_[0].empty())
        return true;
    return false;
}


std::string StringList::toString(bool* ok) const
{
    if (ok) *ok = true;
    std::ostringstream ss;
    for (size_t i = 0; i < values_.size(); ++i) {
        ss << values_.at(i);
        if (i < values_.size() - 1)
            ss << delimiter_;
    }
    return ss.str();
}


void StringList::set(const std::string& s, bool* ok)
{
    if (ok) *ok = true;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimiter_)) {
        values_.push_back(token);
    }
}



size_t StringList::size() const
{
    return values_.size();
}

void StringList::clear()
{
    values_.clear();
}

void StringList::setDelimiter(char d)
{
    delimiter_ = d;
}

std::string StringList::at(size_t i) const
{
    return values_.at(i);
}

std::string StringList::operator[](size_t i) const
{
    return values_[i];
}

StringList& StringList::operator<<(const std::string& s)
{
    values_.push_back(s);
    return *this;
}


bool StringList::isEqual(const StringList& other) const
{
    if (values_.size() != other.values_.size())
        return false;
    for (size_t i = 0; i < values_.size(); ++i) {
        if (std::string(values_.at(i)) != std::string(other.values_.at(i)))
            return false;
    }
    return true;
}

bool StringList::operator==(const StringList& other) const
{
    return isEqual(other);
}

bool StringList::operator!=(const StringList& other) const
{
    return !isEqual(other);
}




} // namespace oskar


