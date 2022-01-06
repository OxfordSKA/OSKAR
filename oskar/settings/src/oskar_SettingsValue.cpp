/*
 * Copyright (c) 2015-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "settings/oskar_SettingsValue.h"
#include "settings/oskar_settings_utility_string.h"
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

namespace oskar {

SettingsValue::SettingsValue()
{
}

SettingsValue::~SettingsValue()
{
}

SettingsValue& SettingsValue::operator=(const SettingsValue& other)
{
    if (this == &other) return *this;
    value_ = other.value_;
    return *this;
}

void SettingsValue::operator=(const value_t& other)
{
    value_ = other;
}

SettingsValue::TypeId SettingsValue::type() const
{
    return value_.is_singular() ? UNDEF :
                    static_cast<SettingsValue::TypeId>(value_.which());
}

const char* SettingsValue::type_name(SettingsValue::TypeId type)
{
    switch (type)
    {
        case UNDEF: return "Undef";
        case BOOL: return "Bool";
        case DATE_TIME: return "DateTime";
        case DOUBLE: return "Double";
        case DOUBLE_LIST: return "DoubleList";
        case DOUBLE_RANGE: return "DoubleRange";
        case DOUBLE_RANGE_EXT: return "DoubleRangeExt";
        case INPUT_DIRECTORY: return "InputDirectory";
        case INPUT_FILE: return "InputFile";
        case INPUT_FILE_LIST: return "InputFileList";
        case INT: return "Int";
        case INT_LIST: return "IntList";
        case INT_LIST_EXT: return "IntListExt";
        case INT_POSITIVE: return "IntPositive";
        case INT_RANGE: return "IntRange";
        case INT_RANGE_EXT: return "IntRangeExt";
        case OPTION_LIST: return "OptionList";
        case OUTPUT_FILE: return "OutputFile";
        case RANDOM_SEED: return "RandomSeed";
        case STRING: return "String";
        case STRING_LIST: return "StringList";
        case TIME: return "Time";
        case UNSIGNED_DOUBLE: return "UnsignedDouble";
        case UNSIGNED_INT: return "UnsignedInt";
        default: return "Undef";
    }
}

const char* SettingsValue::type_name() const
{
    return type_name(type());
}


SettingsValue::TypeId SettingsValue::type_id(const char* type_name)
{
    string t = type_name ?
            oskar_settings_utility_string_to_upper(type_name) : string();
    if (t == "BOOL") { return BOOL; }
    else if (t == "DATETIME") { return DATE_TIME; }
    else if (t == "DOUBLE") { return DOUBLE; }
    else if (t == "DOUBLELIST") { return DOUBLE_LIST; }
    else if (t == "DOUBLERANGE") { return DOUBLE_RANGE; }
    else if (t == "DOUBLERANGEEXT") { return DOUBLE_RANGE_EXT; }
    else if (t == "INPUTDIRECTORY") { return INPUT_DIRECTORY; }
    else if (t == "INPUTFILE") { return INPUT_FILE; }
    else if (t == "INPUTFILELIST") { return INPUT_FILE_LIST; }
    else if (t == "INPUTDIRECTORY") { return INPUT_DIRECTORY; }
    else if (t == "INT") { return INT; }
    else if (t == "INTLIST") { return INT_LIST; }
    else if (t == "INTLISTEXT") { return INT_LIST_EXT; }
    else if (t == "INTPOSITIVE") { return INT_POSITIVE; }
    else if (t == "INTRANGE") { return INT_RANGE; }
    else if (t == "INTRANGEEXT") { return INT_RANGE_EXT; }
    else if (t == "OPTIONLIST") { return OPTION_LIST; }
    else if (t == "OUTPUTFILE") { return OUTPUT_FILE; }
    else if (t == "RANDOMSEED") { return RANDOM_SEED; }
    else if (t == "STRING") { return STRING; }
    else if (t == "STRINGLIST") { return STRING_LIST; }
    else if (t == "TIME") { return TIME; }
    else if (t == "UNSIGNEDDOUBLE") { return UNSIGNED_DOUBLE; }
    else if (t == "UNSIGNEDINT" || t == "UINT") { return UNSIGNED_INT; }
    return UNDEF;
}

bool SettingsValue::init(const char* type, const char* param)
{
    TypeId id = type_id(type);
    if (id == UNDEF) {
        cerr << "ERROR: Failed to initialise settings value, undefined type." << endl;
        return false;
    }
    create(id);
    AbstractSettingsType* t = get(id);
    string par = param ? string(param) : string();
    return t ? t->init(par.c_str()) : false;
}

bool SettingsValue::set_default(const char* value)
{
    AbstractSettingsType* t = get(type());
    string val = value ? string(value) : string();
    return t ? t->set_default(val.c_str()) : false;
}

const char* SettingsValue::get_value() const
{
    const AbstractSettingsType* t =
            const_cast<SettingsValue*>(this)->get(type());
    // Note: The type would not exist if there is no default and no default init
    return t ? t->get_value() : "";
}

const char* SettingsValue::get_default() const
{
    const AbstractSettingsType* t =
            const_cast<SettingsValue*>(this)->get(type());
    return t ? t->get_default() : "";
}

bool SettingsValue::set_value(const char* value)
{
    AbstractSettingsType* t = get(type());
    string val = value ? string(value) : string();
    return t ? t->set_value(val.c_str()) : false;
}

bool SettingsValue::is_default() const
{
    const AbstractSettingsType* t =
            const_cast<SettingsValue*>(this)->get(type());
    return t ? t->is_default() : true;
}

bool SettingsValue::is_set() const
{
    if (value_.is_singular()) return false;
    return !is_default();
}

// Conversions to intrinsic types
double SettingsValue::to_double(bool& ok) const
{
    using ttl::var::get;
    ok = true;
    if (value_.is_singular()) {
        ok = false;
        return 0.0;
    }
    switch (value_.which())
    {
        case DOUBLE:
            return get<Double>(value_).value();
        case UNSIGNED_DOUBLE:
            return get<UnsignedDouble>(value_).value();
        case DOUBLE_RANGE:
            return get<DoubleRange>(value_).value();
        case DOUBLE_RANGE_EXT:
            return get<DoubleRangeExt>(value_).value();
        case DOUBLE_LIST:
        {
            if (get<DoubleList>(value_).size() < 1)
            {
                ok = false;
                return 0.0;
            }
            return get<DoubleList>(value_).values()[0];
        }
        case TIME:
            return get<Time>(value_).to_seconds();
        case DATE_TIME:
            return get<DateTime>(value_).to_mjd();
        default:
            ok = false;
            break;
    };
    return 0.0;
}

int SettingsValue::to_int(bool& ok) const
{
    using ttl::var::get;
    ok = true;
    if (value_.is_singular()) {
        ok = false;
        return 0;
    }
    switch (value_.which())
    {
        case INT:
            return get<Int>(value_).value();
        case INT_RANGE:
            return get<IntRange>(value_).value();
        case INT_RANGE_EXT:
            return get<IntRangeExt>(value_).value();
        case INT_POSITIVE:
            return get<IntPositive>(value_).value();
        case RANDOM_SEED:
            return get<RandomSeed>(value_).value();
        case UNSIGNED_INT:
            return get<UnsignedInt>(value_).value();
        case INT_LIST:
        {
            if (get<IntList>(value_).size() < 1)
            {
                ok = false;
                return 0;
            }
            return get<IntList>(value_).values()[0];
        }
        case BOOL:
            return static_cast<int>(get<Bool>(value_).value());
        default:
            ok = false;
            break;
    };
    return 0;
}

unsigned int SettingsValue::to_unsigned(bool& ok) const
{
    ok = true;
    if (value_.is_singular()) {
        ok = false;
        return 0u;
    }
    switch (value_.which())
    {
        case UNSIGNED_INT:
            return ttl::var::get<UnsignedInt>(value_).value();
        default:
            ok = false;
            break;
    };
    return 0u;
}

const char* SettingsValue::to_string() const
{
    return get_value();
}

const char* const* SettingsValue::to_string_list(int* size, bool& ok) const
{
    ok = false;
    if (value_.is_singular()) return 0;
    switch (value_.which())
    {
        case STRING_LIST:
            ok = true;
            *size = ttl::var::get<StringList>(value_).size();
            if (*size > 0) return ttl::var::get<StringList>(value_).values();
            break;
        case INPUT_FILE_LIST:
            ok = true;
            *size = ttl::var::get<InputFileList>(value_).size();
            if (*size > 0) return ttl::var::get<InputFileList>(value_).values();
            break;
        default:
            break;
    };
    return 0;
}

const int* SettingsValue::to_int_list(int* size, bool& ok) const
{
    ok = false;
    if (value_.is_singular()) return 0;
    switch (value_.which())
    {
        case INT_LIST:
            ok = true;
            *size = ttl::var::get<IntList>(value_).size();
            if (*size > 0) return ttl::var::get<IntList>(value_).values();
            break;
        case INT_LIST_EXT:
            ok = true;
            *size = ttl::var::get<IntListExt>(value_).size();
            if (*size > 0) return ttl::var::get<IntListExt>(value_).values();
            break;
        default:
            break;
    };
    return 0;
}

const double* SettingsValue::to_double_list(int* size, bool& ok) const
{
    ok = false;
    if (value_.is_singular()) return 0;
    switch (value_.which())
    {
        case DOUBLE_LIST:
            ok = true;
            *size = ttl::var::get<DoubleList>(value_).size();
            if (*size > 0) return ttl::var::get<DoubleList>(value_).values();
            break;
        default:
            break;
    };
    return 0;
}


bool SettingsValue::operator==(const SettingsValue& other) const
{
    using ttl::var::get;
    switch (type())
    {
        case BOOL:
            return get<Bool>(value_) == get<Bool>(other.value_);
        case DATE_TIME:
            return get<DateTime>(value_) == get<DateTime>(other.value_);
        case DOUBLE:
            return get<Double>(value_) == get<Double>(other.value_);
        case DOUBLE_LIST:
            return get<DoubleList>(value_) == get<DoubleList>(other.value_);
        case DOUBLE_RANGE:
            return get<DoubleRange>(value_) == get<DoubleRange>(other.value_);
        case DOUBLE_RANGE_EXT:
            return get<DoubleRangeExt>(value_) == get<DoubleRangeExt>(other.value_);
        case INPUT_DIRECTORY:
            return get<InputDirectory>(value_) == get<InputDirectory>(other.value_);
        case INPUT_FILE:
            return get<InputFile>(value_) == get<InputFile>(other.value_);
        case INPUT_FILE_LIST:
            return get<InputFileList>(value_) == get<InputFileList>(other.value_);
        case INT:
            return get<Int>(value_) == get<Int>(other.value_);
        case INT_LIST:
            return get<IntList>(value_) == get<IntList>(other.value_);
        case INT_LIST_EXT:
            return get<IntListExt>(value_) == get<IntListExt>(other.value_);
        case INT_POSITIVE:
            return get<IntPositive>(value_) == get<IntPositive>(other.value_);
        case INT_RANGE:
            return get<IntRange>(value_) == get<IntRange>(other.value_);
        case INT_RANGE_EXT:
            return get<IntRangeExt>(value_) == get<IntRangeExt>(other.value_);
        case OPTION_LIST:
            return get<OptionList>(value_) == get<OptionList>(other.value_);
        case OUTPUT_FILE:
            return get<OutputFile>(value_) == get<OutputFile>(other.value_);
        case RANDOM_SEED:
            return get<RandomSeed>(value_) == get<RandomSeed>(other.value_);
        case STRING:
            return get<String>(value_) == get<String>(other.value_);
        case STRING_LIST:
            return get<StringList>(value_) == get<StringList>(other.value_);
        case TIME:
            return get<Time>(value_) == get<Time>(other.value_);
        case UNSIGNED_DOUBLE:
            return get<UnsignedDouble>(value_) == get<UnsignedDouble>(other.value_);
        case UNSIGNED_INT:
            return get<UnsignedInt>(value_) == get<UnsignedInt>(other.value_);
        default:
            return false;
    };
    return false;
}

bool SettingsValue::operator>(const SettingsValue& other) const
{
    using ttl::var::get;
    switch (type())
    {
        case BOOL:
            return get<Bool>(value_) > get<Bool>(other.value_);
        case DATE_TIME:
            return get<DateTime>(value_) > get<DateTime>(other.value_);
        case DOUBLE:
            return get<Double>(value_) > get<Double>(other.value_);
        case DOUBLE_LIST:
            return get<DoubleList>(value_) > get<DoubleList>(other.value_);
        case DOUBLE_RANGE:
            return get<DoubleRange>(value_) > get<DoubleRange>(other.value_);
        case DOUBLE_RANGE_EXT:
            return get<DoubleRangeExt>(value_) > get<DoubleRangeExt>(other.value_);
        case INPUT_DIRECTORY:
            return get<InputDirectory>(value_) > get<InputDirectory>(other.value_);
        case INPUT_FILE:
            return get<InputFile>(value_) > get<InputFile>(other.value_);
        case INPUT_FILE_LIST:
            return get<InputFileList>(value_) > get<InputFileList>(other.value_);
        case INT:
            return get<Int>(value_) > get<Int>(other.value_);
        case INT_LIST:
            return get<IntList>(value_) > get<IntList>(other.value_);
        case INT_LIST_EXT:
            return get<IntListExt>(value_) > get<IntListExt>(other.value_);
        case INT_POSITIVE:
            return get<IntPositive>(value_) > get<IntPositive>(other.value_);
        case INT_RANGE:
            return get<IntRange>(value_) > get<IntRange>(other.value_);
        case INT_RANGE_EXT:
            return get<IntRangeExt>(value_) > get<IntRangeExt>(other.value_);
        case OPTION_LIST:
            return get<OptionList>(value_) > get<OptionList>(other.value_);
        case OUTPUT_FILE:
            return get<OutputFile>(value_) > get<OutputFile>(other.value_);
        case RANDOM_SEED:
            return get<RandomSeed>(value_) > get<RandomSeed>(other.value_);
        case STRING:
            return get<String>(value_) > get<String>(other.value_);
        case STRING_LIST:
            return get<StringList>(value_) > get<StringList>(other.value_);
        case TIME:
            return get<Time>(value_) > get<Time>(other.value_);
        case UNSIGNED_DOUBLE:
            return get<UnsignedDouble>(value_) > get<UnsignedDouble>(other.value_);
        case UNSIGNED_INT:
            return get<UnsignedInt>(value_) > get<UnsignedInt>(other.value_);
        default:
            return false;
    };
    return false;
}

bool SettingsValue::operator!=(const SettingsValue& other) const
{
    return !(*this == other);
}

bool SettingsValue::operator>=(const SettingsValue& other) const
{
    return (*this > other || *this == other);
}

bool SettingsValue::operator<(const SettingsValue& other) const
{
    return !(*this > other) && !(*this == other);
}

bool SettingsValue::operator<=(const SettingsValue& other) const
{
    return !(*this > other);
}

void SettingsValue::create(TypeId type)
{
    switch (type)
    {
        case BOOL:             value_ = Bool(); return;
        case DATE_TIME:        value_ = DateTime(); return;
        case DOUBLE:           value_ = Double(); return;
        case DOUBLE_LIST:      value_ = DoubleList(); return;
        case DOUBLE_RANGE:     value_ = DoubleRange(); return;
        case DOUBLE_RANGE_EXT: value_ = DoubleRangeExt(); return;
        case INPUT_DIRECTORY:  value_ = InputDirectory(); return;
        case INPUT_FILE:       value_ = InputFile(); return;
        case INPUT_FILE_LIST:  value_ = InputFileList(); return;
        case INT:              value_ = Int(); return;
        case INT_LIST:         value_ = IntList(); return;
        case INT_LIST_EXT:     value_ = IntListExt(); return;
        case INT_POSITIVE:     value_ = IntPositive(); return;
        case INT_RANGE:        value_ = IntRange(); return;
        case INT_RANGE_EXT:    value_ = IntRangeExt(); return;
        case OPTION_LIST:      value_ = OptionList(); return;
        case OUTPUT_FILE:      value_ = OutputFile(); return;
        case RANDOM_SEED:      value_ = RandomSeed(); return;
        case STRING:           value_ = String(); return;
        case STRING_LIST:      value_ = StringList(); return;
        case TIME:             value_ = Time(); return;
        case UNSIGNED_DOUBLE:  value_ = UnsignedDouble(); return;
        case UNSIGNED_INT:     value_ = UnsignedInt(); return;
        default:               return;
    }
}

AbstractSettingsType* SettingsValue::get(TypeId type)
{
    using ttl::var::get;
    switch (type)
    {
        case BOOL:             return &get<Bool>(value_);
        case DATE_TIME:        return &get<DateTime>(value_);
        case DOUBLE:           return &get<Double>(value_);
        case DOUBLE_LIST:      return &get<DoubleList>(value_);
        case DOUBLE_RANGE:     return &get<DoubleRange>(value_);
        case DOUBLE_RANGE_EXT: return &get<DoubleRangeExt>(value_);
        case INPUT_DIRECTORY:  return &get<InputDirectory>(value_);
        case INPUT_FILE:       return &get<InputFile>(value_);
        case INPUT_FILE_LIST:  return &get<InputFileList>(value_);
        case INT:              return &get<Int>(value_);
        case INT_LIST:         return &get<IntList>(value_);
        case INT_LIST_EXT:     return &get<IntListExt>(value_);
        case INT_POSITIVE:     return &get<IntPositive>(value_);
        case INT_RANGE:        return &get<IntRange>(value_);
        case INT_RANGE_EXT:    return &get<IntRangeExt>(value_);
        case OPTION_LIST:      return &get<OptionList>(value_);
        case OUTPUT_FILE:      return &get<OutputFile>(value_);
        case RANDOM_SEED:      return &get<RandomSeed>(value_);
        case STRING:           return &get<String>(value_);
        case STRING_LIST:      return &get<StringList>(value_);
        case TIME:             return &get<Time>(value_);
        case UNSIGNED_DOUBLE:  return &get<UnsignedDouble>(value_);
        case UNSIGNED_INT:     return &get<UnsignedInt>(value_);
        default:               return 0;
    }
}

} // namespace oskar
