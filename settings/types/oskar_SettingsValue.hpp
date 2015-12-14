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

#ifndef OSKAR_SETTINGS_VALUE_HPP_
#define OSKAR_SETTINGS_VALUE_HPP_

#ifdef __cplusplus

#include <oskar_settings_types.hpp>
#include <ttl/var/variant.hpp>

namespace oskar {

class SettingsValue
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
    static SettingsValue::TypeId get_type_id(const std::string& type_name);
    static const char* type_name(SettingsValue::TypeId type);
    std::string type_name() const;

    template <typename T> T& get();
    template <typename T> const T& get() const;
    template <typename T1, typename T2> bool set(const T2& v);
    template <typename T> std::string value() const;

    /* Basic string methods for interfacing with the type. */
    bool init(const std::string& type, const std::string& param);
    bool init(TypeId type, const std::string& param);
    bool set_default(const std::string& value);
    bool set_value(const std::string&);
    std::string get_value() const;
    std::string get_default() const;
    bool is_default() const;
    bool is_set() const;

    /* Conversions to intrinsic types */
    double to_double(bool& ok) const;
    int to_int(bool& ok) const;
    unsigned int to_unsigned(bool& ok) const;
    std::string to_string() const;
    std::vector<std::string> to_string_list(bool& ok) const;
    std::vector<int> to_int_list(bool& ok) const;
    std::vector<double> to_double_list(bool& ok) const;

    bool operator==(const SettingsValue& other) const;
    bool operator!=(const SettingsValue& other) const;
    bool operator>(const SettingsValue& other) const;
    bool operator>=(const SettingsValue& other) const;
    bool operator<(const SettingsValue& other) const;
    bool operator<=(const SettingsValue& other) const;

    operator std::string() const;

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

template <typename T1, typename T2>
bool SettingsValue::set(const T2& v)
{
    return ttl::var::get<T1>(value_).set_value(v);
}

template <typename T>
std::string SettingsValue::value() const
{
    return ttl::var::get<T>(value_).get_value();
}

} /* namespace oskar */

#endif /* __cplusplus */

#ifdef __cplusplus
extern "C" {
#endif

/* C interface. */
struct oskar_SettingsValue;
#ifndef OSKAR_SETTINGS_VALUE_TYPEDEF_
#define OSKAR_SETTINGS_VALUE_TYPEDEF_
typedef struct oskar_SettingsValue oskar_SettingsValue;
#endif /* OSKAR_SETTINGS_VALUE_TYPEDEF_ */

enum OSKAR_SETTINGS_ERRORS
{
    OSKAR_ERR_SETTINGS_NO_VALUE = -200,
    OSKAR_ERR_SETTINGS_INT_CONVERSION_FAIL = -201,
    OSKAR_ERR_SETTINGS_UNSIGNED_INT_CONVERSION_FAIL = -202,
    OSKAR_ERR_SETTINGS_DOUBLE_CONVERSION_FAIL = -203,
    OSKAR_ERR_SETTINGS_INT_LIST_CONVERSION_FAIL = -204,
    OSKAR_ERR_SETTINGS_DOUBLE_LIST_CONVERSION_FAIL = -205,
    OSKAR_ERR_SETTINGS_STRING_LIST_CONVERSION_FAIL = -206,
    OSKAR_ERR_SETTINGS_LOAD = -207,
    OSKAR_ERR_SETTINGS_SAVE = -208
};

/* Basic string methods for interfacing with the type. */
int oskar_settings_value_init(oskar_SettingsValue* v, const char* type,
        const char* param);
int oskar_settings_value_set_default(oskar_SettingsValue* v, const char* value);
int oskar_settings_value_set(oskar_SettingsValue* v, const char* value);
char* oskar_settings_value_string(const oskar_SettingsValue* v, int* status);
int oskar_settings_value_is_default(const oskar_SettingsValue* v);
int oskar_settings_value_starts_with(const oskar_SettingsValue* v,
        const char* str, int* status);
char oskar_settings_value_first_letter(const oskar_SettingsValue* v,
        int* status);

/* Conversions to intrinsic types. */
double oskar_settings_value_to_double(const oskar_SettingsValue* v, int* status);
int oskar_settings_value_to_int(const oskar_SettingsValue* v, int* status);
unsigned int oskar_settings_value_to_unsigned(const oskar_SettingsValue* v,
        int* status);
char** oskar_settings_value_to_string_list(const oskar_SettingsValue* v,
        int* num, int* status);
int* oskar_settings_value_to_int_list(const oskar_SettingsValue* v,
        int* num, int* status);
double* oskar_settings_value_to_double_list(const oskar_SettingsValue* v,
        int* num, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_SETTINGS_VALUE_HPP_ */
