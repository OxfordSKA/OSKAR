/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SETTINGS_VALUE_HPP_
#define OSKAR_SETTINGS_VALUE_HPP_

#include <settings/oskar_settings_macros.h>

#ifdef __cplusplus

#include <settings/oskar_settings_types.h>
#include <settings/extern/ttl/var/variant.hpp>

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
    OSKAR_SETTINGS_EXPORT SettingsValue();
    OSKAR_SETTINGS_EXPORT virtual ~SettingsValue();

public:
    OSKAR_SETTINGS_EXPORT void operator=(const value_t& other);
    OSKAR_SETTINGS_EXPORT SettingsValue& operator=(const SettingsValue& other);

public:
    OSKAR_SETTINGS_EXPORT SettingsValue::TypeId type() const;
    OSKAR_SETTINGS_EXPORT static SettingsValue::TypeId type_id(const char* type_name);
    OSKAR_SETTINGS_EXPORT static const char* type_name(SettingsValue::TypeId type);
    OSKAR_SETTINGS_EXPORT const char* type_name() const;

    template <typename T> T& get();
    template <typename T> const T& get() const;

    /* Basic string methods for interfacing with the type. */
    OSKAR_SETTINGS_EXPORT bool init(const char* type, const char* param);
    OSKAR_SETTINGS_EXPORT bool set_default(const char* value);
    OSKAR_SETTINGS_EXPORT bool set_value(const char* value);
    OSKAR_SETTINGS_EXPORT const char* get_value() const;
    OSKAR_SETTINGS_EXPORT const char* get_default() const;
    OSKAR_SETTINGS_EXPORT bool is_default() const;
    OSKAR_SETTINGS_EXPORT bool is_set() const;

    /* Conversions to intrinsic types */
    OSKAR_SETTINGS_EXPORT double to_double(bool& ok) const;
    OSKAR_SETTINGS_EXPORT int to_int(bool& ok) const;
    OSKAR_SETTINGS_EXPORT unsigned int to_unsigned(bool& ok) const;
    OSKAR_SETTINGS_EXPORT const char* to_string() const;
    OSKAR_SETTINGS_EXPORT const char* const* to_string_list(int* size,
            bool& ok) const;
    OSKAR_SETTINGS_EXPORT const int* to_int_list(int* size, bool& ok) const;
    OSKAR_SETTINGS_EXPORT const double* to_double_list(int* size,
            bool& ok) const;

    OSKAR_SETTINGS_EXPORT bool operator==(const SettingsValue& other) const;
    OSKAR_SETTINGS_EXPORT bool operator!=(const SettingsValue& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>(const SettingsValue& other) const;
    OSKAR_SETTINGS_EXPORT bool operator>=(const SettingsValue& other) const;
    OSKAR_SETTINGS_EXPORT bool operator<(const SettingsValue& other) const;
    OSKAR_SETTINGS_EXPORT bool operator<=(const SettingsValue& other) const;

private:
    void create(SettingsValue::TypeId type);
    AbstractSettingsType* get(SettingsValue::TypeId type);
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
