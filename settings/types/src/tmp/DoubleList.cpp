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

#include <DoubleList.hpp>
#include <sstream>
#include <boost/lexical_cast.hpp>

namespace oskar {

DoubleList::DoubleList()
{
}

DoubleList::DoubleList(const char* s, char delimiter)
{
    if (s) fromStr(s, delimiter);
}

DoubleList::~DoubleList()
{
}

bool DoubleList::isSet() const
{
    return !values_.empty();
}

std::string DoubleList::toString(bool* ok) const
{
    if (ok) *ok = true; // TODO set this properly!
    return std::string(toStr());
}

void DoubleList::set(const std::string& s, bool* ok)
{
    fromStr(s.c_str(),',', ok);
}


size_t DoubleList::size() const
{
    return values_.size();
}

void DoubleList::clear()
{
    values_.clear();
}

const char* DoubleList::toStr(char delimiter) const
{
    if (values_.empty())
        return "";
    else {
        std::ostringstream ss;
        for (size_t i = 0; i < values_.size(); ++i) {
            ss << values_.at(i);
            if (i < values_.size() - 1)
                ss << delimiter;
        }
        return ss.str().c_str();
    }
}

void DoubleList::fromStr(const char* s, char delimeter, bool* ok)
{
    if (ok) *ok = true;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, delimeter)) {
        double d = 0;
        try {
            d = boost::lexical_cast<double>(token);
        }
        catch (boost::bad_lexical_cast&) {
            if (ok) *ok = false;
            continue;
        }
        values_.push_back(d);
    }
}


double DoubleList::at(size_t i) const
{
    return values_.at(i);
}

double DoubleList::operator[](size_t i) const
{
    return values_[i];
}

double& DoubleList::operator[](size_t i)
{
    return values_[i];
}

DoubleList& DoubleList::operator<<(double i)
{
    values_.push_back(i);
    return *this;
}


bool DoubleList::isEqual(const DoubleList& other) const
{
    if (values_.size() != other.values_.size())
        return false;
    for (size_t i = 0; i < values_.size(); ++i) {
        if (values_.at(i) != other.values_.at(i))
            return false;
    }
    return true;
}

bool DoubleList::operator==(const DoubleList& other) const
{
    return isEqual(other);
}

bool DoubleList::operator!=(const DoubleList& other) const
{
    return !isEqual(other);
}


} // namespace oskar
 
