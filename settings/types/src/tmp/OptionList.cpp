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

#include <OptionList.hpp>
#include <sstream>
#include <algorithm>
#include <functional>
#include <cctype>
#include <locale>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/lexical_cast.hpp>


namespace oskar {

// trim from start
static inline std::string &ltrim(std::string &s) {
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), std::not1(std::ptr_fun<int, int>(std::isspace))));
    return s;
}
// trim from end
static inline std::string &rtrim(std::string &s) {
    s.erase(std::find_if(s.rbegin(), s.rend(), std::not1(std::ptr_fun<int, int>(std::isspace))).base(), s.end());
    return s;
}
// trim from both ends
static inline std::string &trim(std::string &s) {
    return ltrim(rtrim(s));
}



OptionList::OptionList()
{ }

OptionList::OptionList(const std::vector<std::string>& options,
        const std::string& value)
: options_(options)
{
    set(value);
}

OptionList::OptionList(const char* optionsCSV, const std::string& value)
{
    char delimeter = ',';
    std::istringstream ss(optionsCSV);
    std::string token;
    while (std::getline(ss, token, delimeter)) {
        options_.push_back(trim(token));
    }
    set(value);
}

OptionList::OptionList(const std::vector<std::string>& options, int valueIndex)
: options_(options)
{
    if (valueIndex < (int)options_.size())
        set(options_[valueIndex]);
}


OptionList::OptionList(const char* optionsCSV, int valueIndex)
{
    char delimeter = ',';
    std::istringstream ss(optionsCSV);
    std::string token;
    while (std::getline(ss, token, delimeter)) {
        options_.push_back(trim(token));
    }
    if (valueIndex < (int)options_.size())
        set(options_[valueIndex]);
}


OptionList::~OptionList()
{
}

std::string OptionList::toString(bool* ok) const
{
    if (ok) *ok = true; // FIXME handle ok properly!!
    return value();
}

std::string OptionList::value() const
{
    return value_;
}

std::vector<std::string> OptionList::options() const
{
    return options_;
}

int OptionList::num_options() const
{ return (int)options_.size(); }

std::string OptionList::option(int at) const
{
    if (at < (int)options_.size())
        return options_[at];
    return std::string();
}

int OptionList::valueIndex(bool* ok) const
{
    if (ok) *ok = false;
    for (size_t i = 0; i < options_.size(); ++i) {
        if (options_[i] == value_) {
            if (ok) *ok = true;
            return i;
        }
    }
    return -1;
}

bool OptionList::isSet() const
{
    return (!value_.empty());
}

void OptionList::set(const std::string& s, bool* ok)
{
    // Settings the option should be by minimal match using starts with
    // http://goo.gl/6NJLgf
    if (ok) *ok = false;
    if (s.empty()) return;

    for (size_t i = 0; i < options_.size(); ++i) {
        if (boost::starts_with(options_[i], s)) {
            if (ok) *ok = true;
            value_ = options_[i];
            return;
        }
    }
    return;
}

void OptionList::fromString(const std::string& s, bool* ok)
{
    this->set(s, ok);
}

} // namespace oskar
 
