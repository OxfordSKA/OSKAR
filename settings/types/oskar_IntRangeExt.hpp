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

#ifndef OSKAR_SETTINGS_TYPE_INTRANGEEXT_HPP_
#define OSKAR_SETTINGS_TYPE_INTRANGEEXT_HPP_

#include <oskar_AbstractSettingsType.hpp>
#include <ttl/var/variant.hpp>
#include <string>

namespace oskar {

class IntRangeExt : public AbstractSettingsType
{
public:
    typedef ttl::var::variant<int, std::string> Value;

    IntRangeExt();
    virtual ~IntRangeExt();

    bool init(const std::string& s);
    bool set_default(const std::string &value);
    std::string get_default() const;
    bool set_value(const std::string& value);
    std::string get_value() const;
    bool is_default() const;

    bool set_value(int i);
    bool set_default(int i);

    // FIXME(BM) handle string types.
    int value() const { return ttl::var::get<int>(value_); }
    // FIXME(BM) handle string types.
    int default_value() const { return ttl::var::get<int>(default_); }
    int min() const { return min_; }
    int max() const { return max_; }
    std::string ext_min() const { return ext_min_; }
    std::string ext_max() const { return ext_max_; }
    bool is_max() const {
        return (value_.which() == STRING &&
                        ttl::var::get<std::string>(value_) == ext_max_);
    }
    bool is_min() const {
        return (value_.which() == STRING &&
                        ttl::var::get<std::string>(value_) == ext_min_);
    }

    bool operator==(const IntRangeExt& other) const;
    bool operator>(const IntRangeExt& other) const;

private:
    bool from_string_(Value& value, const std::string& s) const;
    bool from_int_(Value& value, int i) const;
    std::string to_string_(const Value& value) const;

    int min_, max_;
    std::string ext_min_, ext_max_;
    enum value_types { INT, STRING };
    Value default_, value_;
};

} // namespace oskar
#endif /* OSKAR_SETTINGS_TYPE_INTRANGEEXT_HPP_ */
