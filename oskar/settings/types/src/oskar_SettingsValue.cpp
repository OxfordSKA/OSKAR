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

#include <oskar_SettingsValue.hpp>
#include <oskar_settings_utility_string.hpp>
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

void SettingsValue::operator=(const SettingsValue& other)
{
    value_ = other.value_;
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
        default: return 0;
    }
}

string SettingsValue::type_name() const
{
    const char* t = type_name(type());
    return t ? string(t) : "Undef";
}


SettingsValue::TypeId SettingsValue::get_type_id(const string& type_name)
{
    string t = oskar_settings_utility_string_to_upper(type_name);
    if (t == "BOOL") return BOOL;
    else if (t == "DATETIME") return DATE_TIME;
    else if (t == "DOUBLE") return DOUBLE;
    else if (t == "DOUBLELIST") return DOUBLE_LIST;
    else if (t == "DOUBLERANGE") return DOUBLE_RANGE;
    else if (t == "DOUBLERANGEEXT") return DOUBLE_RANGE_EXT;
    else if (t == "INPUTDIRECTORY") return INPUT_DIRECTORY;
    else if (t == "INPUTFILE") return INPUT_FILE;
    else if (t == "INPUTFILELIST") return INPUT_FILE_LIST;
    else if (t == "INPUTDIRECTORY") return INPUT_DIRECTORY;
    else if (t == "INT") return INT;
    else if (t == "INTLIST") return INT_LIST;
    else if (t == "INTLISTEXT") return INT_LIST_EXT;
    else if (t == "INTPOSITIVE") return INT_POSITIVE;
    else if (t == "INTRANGE") return INT_RANGE;
    else if (t == "INTRANGEEXT") return INT_RANGE_EXT;
    else if (t == "OPTIONLIST") return OPTION_LIST;
    else if (t == "OUTPUTFILE") return OUTPUT_FILE;
    else if (t == "RANDOMSEED") return RANDOM_SEED;
    else if (t == "STRING") return STRING;
    else if (t == "STRINGLIST") return STRING_LIST;
    else if (t == "TIME") return TIME;
    else if (t == "UNSIGNEDDOUBLE") return UNSIGNED_DOUBLE;
    else if (t == "UNSIGNEDINT" || t == "UINT") return UNSIGNED_INT;
    else return UNDEF;
}

bool SettingsValue::init(const string& type, const string& param)
{
    TypeId type_id = get_type_id(type);
    if (type_id == UNDEF) {
        cerr << "ERROR: Failed to initialise settings value, undefined type." << endl;
    }
    return this->init(type_id, param);
}

bool SettingsValue::init(TypeId type, const string& param)
{
    create_(type);
    AbstractSettingsType* t = get_(type);
    return t ? t->init(param) : false;
}

bool SettingsValue::set_default(const string& value)
{
    AbstractSettingsType* t = get_(type());
    return t ? t->set_default(value) : false;
}

string SettingsValue::get_value() const
{
    const AbstractSettingsType* t =
            const_cast<SettingsValue*>(this)->get_(type());
    // Note: The type would not exist if there is no default and no default init
    return t ? t->get_value() : string();
}

std::string SettingsValue::get_default() const
{
    const AbstractSettingsType* t = 
            const_cast<SettingsValue*>(this)->get_(type());
    return t ? t->get_default() : string();
}

bool SettingsValue::set_value(const string& s)
{
    AbstractSettingsType* t = get_(type());
    return t ? t->set_value(s) : false;
}

bool SettingsValue::is_default() const
{
    const AbstractSettingsType* t =
            const_cast<SettingsValue*>(this)->get_(type());
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
            vector<double> t = get<DoubleList>(value_).values();
            if (t.size() < 1)
            {
                ok = false;
                return 0.0;
            }
            return t[0];
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
            vector<int> t = get<IntList>(value_).values();
            if (t.size() < 1)
            {
                ok = false;
                return 0;
            }
            return t[0];
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

string SettingsValue::to_string() const
{
    return get_value();
}

vector<string> SettingsValue::to_string_list(bool& ok) const
{
    ok = true;
    vector<string> list;
    if (value_.is_singular()) {
        ok = false;
        return list;
    }
    switch (value_.which())
    {
        case STRING_LIST:
            return ttl::var::get<StringList>(value_).values();
        case INPUT_FILE_LIST:
            return ttl::var::get<InputFileList>(value_).values();
        default:
            ok = false;
            break;
    };
    return list;
}

vector<int> SettingsValue::to_int_list(bool& ok) const
{
    ok = true;
    vector<int> list;
    if (value_.is_singular()) {
        ok = false;
        return list;
    }
    switch (value_.which())
    {
        case INT_LIST:
            list = ttl::var::get<IntList>(value_).values();
            break;
        default:
            ok = false;
            break;
    };
    if (list.size() == 0) ok = false;
    return list;
}

vector<double> SettingsValue::to_double_list(bool& ok) const
{
    ok = true;
    vector<double> list;
    if (value_.is_singular()) {
        ok = false;
        return list;
    }
    switch (value_.which())
    {
        case DOUBLE_LIST:
            list = ttl::var::get<DoubleList>(value_).values();
            break;
        default:
            ok = false;
            break;
    };
    if (list.size() == 0) ok = false;
    return list;
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

SettingsValue::operator std::string() const
{
    return to_string();
}

void SettingsValue::create_(TypeId type)
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

AbstractSettingsType* SettingsValue::get_(TypeId type)
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


/* C interface. */
struct oskar_SettingsValue : public oskar::SettingsValue
{
};

int oskar_settings_value_init(oskar_SettingsValue* v, const char* type,
        const char* param)
{
    return (int) v->init(string(type), string(param));
}

int oskar_settings_value_set_default(oskar_SettingsValue* v, const char* value)
{
    return (int) v->set_default(string(value));
}

int oskar_settings_value_set(oskar_SettingsValue* v, const char* value)
{
    return (int) v->set_value(string(value));
}

char* oskar_settings_value_string(const oskar_SettingsValue* v, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    string t = v->get_value();
    char* x = (char*) calloc(1, 1 + t.size());
    strcpy(x, t.c_str());
    return x;
}

int oskar_settings_value_is_default(const oskar_SettingsValue* v)
{
    if (!v) return 0;
    return (int) v->is_default();
}

int oskar_settings_value_starts_with(const oskar_SettingsValue* v,
        const char* str, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    string t1 = v->get_value();
    string t2 = string(str);
    return (int) oskar_settings_utility_string_starts_with(t1, t2, false);
}

char oskar_settings_value_first_letter(const oskar_SettingsValue* v,
        int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return '\0';
    }
    string t1 = v->get_value();
    if (t1.empty())
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return '\0';
    }
    int t = t1[0];
    return (char) toupper(t);
}

/* Conversions to intrinsic types. */
double oskar_settings_value_to_double(const oskar_SettingsValue* v, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0.0;
    }
    bool ok;
    double t = v->to_double(ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_DOUBLE_CONVERSION_FAIL;
    return t;
}

int oskar_settings_value_to_int(const oskar_SettingsValue* v, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    bool ok;
    int t = v->to_int(ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_INT_CONVERSION_FAIL;
    return t;
}

unsigned int oskar_settings_value_to_unsigned(const oskar_SettingsValue* v,
        int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0u;
    }
    bool ok;
    unsigned int t = v->to_unsigned(ok);
    if (!ok) *status = OSKAR_ERR_SETTINGS_UNSIGNED_INT_CONVERSION_FAIL;
    return t;
}

char** oskar_settings_value_to_string_list(const oskar_SettingsValue* v,
        int* num, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    bool ok;
    vector<string> t = v->to_string_list(ok);
    if (!ok)
    {
        *status = OSKAR_ERR_SETTINGS_STRING_LIST_CONVERSION_FAIL;
        return 0;
    }
    size_t n = t.size();
    char** x = (char**) calloc(1, n * sizeof(char*));
    for (size_t i = 0; i < n; ++i)
    {
        x[i] = (char*) calloc(1, 1 + t[i].length());
        strcpy(x[i], t[i].c_str());
    }
    *num = (int) n;
    return x;
}

int* oskar_settings_value_to_int_list(const oskar_SettingsValue* v,
        int* num, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    bool ok;
    vector<int> t = v->to_int_list(ok);
    if (!ok)
    {
        *status = OSKAR_ERR_SETTINGS_INT_LIST_CONVERSION_FAIL;
        return 0;
    }
    size_t n = t.size();
    int* x = (int*) calloc(1, n * sizeof(int));
    for (size_t i = 0; i < n; ++i) x[i] = t[i];
    *num = (int) n;
    return x;
}

double* oskar_settings_value_to_double_list(const oskar_SettingsValue* v,
        int* num, int* status)
{
    if (*status) return 0;
    if (!v)
    {
        *status = OSKAR_ERR_SETTINGS_NO_VALUE;
        return 0;
    }
    bool ok;
    vector<double> t = v->to_double_list(ok);
    if (!ok)
    {
        *status = OSKAR_ERR_SETTINGS_DOUBLE_LIST_CONVERSION_FAIL;
        return 0;
    }
    size_t n = t.size();
    double* x = (double*) calloc(1, n * sizeof(double));
    for (size_t i = 0; i < n; ++i) x[i] = t[i];
    *num = (int) n;
    return x;
}
