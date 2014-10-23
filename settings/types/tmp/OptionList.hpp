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

#ifndef OSKAR_SETTINGS_TYPE_OPTIONLIST_HPP_
#define OSKAR_SETTINGS_TYPE_OPTIONLIST_HPP_

/**
 * @file OptionList.hpp
 */

#include <AbstractType.hpp>
#include <string>
#include <vector>

namespace oskar {

class OptionList : public AbstractType
{
public:
    OptionList();
    OptionList(const std::vector<std::string>& options,
            const std::string& value = std::string());
    OptionList(const char* optionsCSV, const std::string& value = std::string());
    OptionList(const std::vector<std::string>& options, int valueIndex);
    OptionList(const char* optionsCSV, int valueIndex);

    virtual ~OptionList();

    bool isSet() const;
    void set(const std::string& s, bool* ok = 0);
    std::string toString(bool* ok = 0) const;

    std::string value() const;
    std::vector<std::string> options() const;
    int num_options() const;
    std::string option(int at = 0) const;
    // Return the index of the value w.r.t. the allowed options list (0 based).
    int valueIndex(bool* ok = 0) const;
    void fromString(const std::string& s, bool* ok = 0);

private:
    std::vector<std::string> options_;
    std::string value_;
};

}
#endif /* OSKAR_SETTINGS_TYPE_OPTIONLIST_HPP_ */
