/*
 * Copyright (c) 2015-2017, The University of Oxford
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

#ifndef OSKAR_SETTINGS_VALUE_HPP_
#define OSKAR_SETTINGS_VALUE_HPP_

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

#include <settings/oskar_settings_types.h>
#include <settings/extern/ttl/var/variant.hpp>

namespace oskar {

class OSKAR_SETTINGS_EXPORT SettingsValue
{
public:
    enum TypeId {
        UNDEF = -1,
        BOOL,
        DATE_TIME,
        DOUBLE,
        DOUBLE_LIST,
        DOUBLE_RANGE,
        DOUBLE_RANGE_EXT,
        INPUT_DIRECTORY,
        INPUT_FILE,
        INPUT_FILE_LIST,
        INT,
        INT_LIST,
        INT_LIST_EXT,
        INT_POSITIVE,
        INT_RANGE,
        INT_RANGE_EXT,
        OPTION_LIST,
        OUTPUT_FILE,
        RANDOM_SEED,
        STRING,
        STRING_LIST,
        TIME,
        UNSIGNED_DOUBLE,
        UNSIGNED_INT
    };

    typedef ttl::var::variant<
            oskar::Bool,
            oskar::DateTime,
            oskar::Double,
            oskar::DoubleList,
            oskar::DoubleRange,
            oskar::DoubleRangeExt,
            oskar::InputDirectory,
            oskar::InputFile,
            oskar::InputFileList,
            oskar::Int,
            oskar::IntList,
            oskar::IntListExt,
            oskar::IntPositive,
            oskar::IntRange,
            oskar::IntRangeExt,
            oskar::OptionList,
            oskar::OutputFile,
            oskar::RandomSeed,
            oskar::String,
            oskar::StringList,
            oskar::Time,
            oskar::UnsignedDouble,
            oskar::UnsignedInt
            > value_t;

public:
    SettingsValue();
    virtual ~SettingsValue();

public:
    void operator=(const value_t& other);
    void operator=(const SettingsValue& other);

public:
    SettingsValue::TypeId type() const;
    static SettingsValue::TypeId type_id(const char* type_name);
    static const char* type_name(SettingsValue::TypeId type);
    const char* type_name() const;

    template <typename T> T& get();
    template <typename T> const T& get() const;

    /* Basic string methods for interfacing with the type. */
    bool init(const char* type, const char* param);
    bool set_default(const char* value);
    bool set_value(const char* value);
    const char* get_value() const;
    const char* get_default() const;
    bool is_default() const;
    bool is_set() const;

    /* Conversions to intrinsic types */
    double to_double(bool& ok) const;
    int to_int(bool& ok) const;
    unsigned int to_unsigned(bool& ok) const;
    const char* to_string() const;
    const char* const* to_string_list(int* size, bool& ok) const;
    const int* to_int_list(int* size, bool& ok) const;
    const double* to_double_list(int* size, bool& ok) const;

    bool operator==(const SettingsValue& other) const;
    bool operator!=(const SettingsValue& other) const;
    bool operator>(const SettingsValue& other) const;
    bool operator>=(const SettingsValue& other) const;
    bool operator<(const SettingsValue& other) const;
    bool operator<=(const SettingsValue& other) const;

private:
    void create_(SettingsValue::TypeId type);
    AbstractSettingsType* get_(SettingsValue::TypeId type);
    value_t value_;
};

template <typename T>
T& SettingsValue::get()
{
    return ttl::var::get<T>(value_);
}

template <typename T>
const T& SettingsValue::get() const
{
    return ttl::var::get<T>(value_);
}

} /* namespace oskar */

#endif /* __cplusplus */

#endif /* OSKAR_SETTINGS_VALUE_HPP_ */
