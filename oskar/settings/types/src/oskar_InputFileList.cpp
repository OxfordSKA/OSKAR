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

#include <oskar_settings_utility_string.hpp>
#include <sstream>
#include <oskar_InputFileList.hpp>

namespace oskar {

InputFileList::InputFileList()
: delimiter_(',')
{
}

InputFileList::~InputFileList()
{
}

bool InputFileList::init(const std::string& /*s*/)
{
    // TODO(BM) Could use this to set the delimiter ...
    value_.clear();
    default_.clear();
    return true;
}

bool InputFileList::set_default(const std::string& value)
{
    value_.clear();
    default_.clear();
    default_ = oskar_settings_utility_string_get_type_params(value);
    for (unsigned int i = 0; i < default_.size(); ++i) {
        value_.push_back(default_.at(i));
    }
    return true;
}

std::string InputFileList::get_default() const
{
    return to_string_(default_);
}

bool InputFileList::set_value(const std::string& value)
{
    value_.clear();
    value_ = oskar_settings_utility_string_get_type_params(value);
    return true;
}

std::string InputFileList::get_value() const
{
    return to_string_(value_);
}

bool InputFileList::is_default() const
{
    if (default_.size() != value_.size())
        return false;
    for (unsigned int i = 0; i < default_.size(); ++i) {
        if (default_.at(i) != value_.at(i))
            return false;
    }
    return true;
}

bool InputFileList::operator==(const InputFileList& other) const
{
    if (value_.size() != other.value_.size())
        return false;
    for (unsigned int i = 0; i < other.value_.size(); ++i) {
        if (value_.at(i) != other.value_.at(i))
            return false;
    }
    return true;
}

bool InputFileList::operator>(const InputFileList& ) const
{
    return false;
}

std::string InputFileList::to_string_(const std::vector<std::string>& values) const
{
    std::ostringstream ss;
    for (size_t i = 0u; i < values.size(); ++i) {
        ss << values.at(i);
        if (i < values.size() - 1)
            ss << delimiter_;
    }
    return ss.str();
}

} // namespace oskar

