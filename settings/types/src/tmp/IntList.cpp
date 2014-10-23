/*
 * Copyright (c) 2013, The University of Oxford
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

#include <IntList.hpp>
#include <boost/lexical_cast.hpp>

namespace oskar {

IntList::IntList(const std::string& s, char delimiter)
: delimiter_(delimiter)
{
    fromString(s, delimiter);
    if (isText())
        addAllowedString(s);
}

IntList::IntList()
: delimiter_(',')
{
}

IntList::~IntList()
{
}

bool IntList::isSet() const
{
    return !(values_.which() == 0);
}

std::string IntList::toString(bool* ok) const
{
    if (ok) *ok = true;
    if (values_.is_singular()) return std::string();

    if (values_.which() == 0) { // std::vector
        std::ostringstream ss;
        const std::vector<int>& v = VAR::get<std::vector<int> >(values_);
        for (size_t i = 0; i < v.size(); ++i) {
            ss << v.at(i);
            if (i < v.size() - 1)
                ss << delimiter_;
        }
        return ss.str();
    }
    else if (values_.which() == 1) {
        return VAR::get<std::string>(values_);
    }
    else {
        if (ok) *ok = false;
        return std::string();
    }
}

void IntList::set(const std::string& s, bool* ok)
{
    if (ok) *ok = false;
    if (allowedStrings_.size() > 0) {
        for (size_t i = 0; i < allowedStrings_.size(); ++i) {
            if (std::string(s) == allowedStrings_[i]) {
                if (ok) *ok = true;
                values_ = std::string(s);
            }
        }
    }
    else {
        if (ok) *ok = true;
        values_ = std::string(s);
    }
}


size_t IntList::size() const
{
    if (values_.is_singular()) return 0;
    if (values_.which() == 0) { // std::vector
        return VAR::get<std::vector<int> >(values_).size();
    }
    else { // std::vector
        return (VAR::get<std::string>(values_).size() == 0) ? 0 : 1;
    }
}

void IntList::clear()
{
    if (values_.is_singular()) return;

    if (values_.which() == 0) { // std::vector
        return VAR::get<std::vector<int> >(values_).clear();
    }
    else { // std::string
        return VAR::get<std::string>(values_).clear();
    }
}

void IntList::fromString(const std::string& s, char delimeter, bool* ok)
{
    if (ok) *ok = true;
    std::istringstream ss(s);
    std::string token;
    std::vector<std::string> tokens;
    std::vector<int> values;
    while (std::getline(ss, token, delimeter)) {
        int i = 0;
        try {
            i = boost::lexical_cast<int>(token);
        }
        catch (boost::bad_lexical_cast&) {
            if (ok) *ok = false;
            continue;
        }
        tokens.push_back(token);
        values.push_back(i);
    }

    if (values.size() > 0 && values.size() == tokens.size()) {
        values_ = static_cast<std::vector<int> >(values);
    }
    else {
        if (ok) *ok = false;
        if (allowedStrings_.size() > 0) {
            for (size_t i = 0; i < allowedStrings_.size(); ++i) {
                if (allowedStrings_.at(i) == std::string(s)) {
                    if (ok) *ok = true;
                    values_ = std::string(s);
                }
            }
        }
        else {
            if (ok) *ok = true;
            values_ = std::string(s);
        }
    }
}


int IntList::at(size_t i) const
{
    if (values_.which() == 0) // vector.
        return VAR::get<std::vector<int> >(values_).at(i);
    else
        return 0;
}

int IntList::operator[](size_t i) const
{
    if (values_.which() == 0)
        return VAR::get<std::vector<int> >(values_)[i];
    else
        return 0;
}

void IntList::set(size_t index, int i)
{
    if (values_.which() == 0)
        VAR::get<std::vector<int> >(values_)[index] = i;
}


IntList& IntList::operator<<(int i)
{
    std::vector<int> v;
    if (!values_.is_singular() && values_.which() == 0)
        v = VAR::get<std::vector<int> >(values_);
    v.push_back(i);
    values_ = static_cast<std::vector<int> >(v);
    return *this;
}

bool IntList::isValid() const
{
    return !values_.is_singular();
}

bool IntList::isList() const
{
    return (values_.which() == 0);
}

bool IntList::isText() const
{
    return (values_.which() == 1);
}

bool IntList::isEqual(const IntList& other) const
{
    // Both boost::blank
    if (values_.is_singular() && other.values_.is_singular()) {
        return true;
    }

    // Both std::vector<int>
    else if (values_.which() == 0 && other.values_.which() == 0) {
        const std::vector<int> v = VAR::get<std::vector<int> >(values_);
        const std::vector<int> o = VAR::get<std::vector<int> >(other.values_);
        if (v.size() != o.size())
            return false;
        for (size_t i = 0; i < v.size(); ++i) {
            if (v.at(i) != o.at(i))
                return false;
        }
        return true;
    }
    // Both std::string
    else if (values_.which() == 1 && other.values_.which() == 1) {
        std::string v = VAR::get<std::string>(values_);
        std::string o = VAR::get<std::string>(other.values_);
        if (v == o)
            return true;
        else
            return false;
    }
    else
        return false;
}

bool IntList::operator==(const IntList& other) const
{
    return isEqual(other);
}

bool IntList::operator!=(const IntList& other) const
{
    return !isEqual(other);
}

void IntList::addAllowedString(const std::string& s)
{
    allowedStrings_.push_back(s);
}

size_t IntList::numAllowedStrings() const
{
    return allowedStrings_.size();
}

std::string IntList::allowedString(size_t i) const
{
    return allowedStrings_.at(i).c_str();
}

} // namespace oskar

