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

#include <iostream> // For debugging
#include <cassert>
#include <algorithm>
#include <sstream>

#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <ttl/exception.hpp>

#include <cstdio> // for debugging
#include <iostream> // for debugging

#include <oskar_SettingsVariant.hpp>

using namespace std; // for debugging

namespace oskar {

Variant::Variant(MetaTypeId type) : type_(type)
{
#if 0
    int size = boost::mpl::size<all_types>::value;
    cout << size << endl;
#endif
    // mmm call init or not ... ? (if not it makes some of the toXXX()
    // functions not work as they cant cast to a singular)
    // if true the isSet() functions may fail.
    //init_(type);
}
Variant::Variant(const Variant& other)
: type_(other.type_), value_(other.value_)
{ }
Variant::Variant(MetaTypeId type, const std::string& s) : type_(type)
{ fromString(s); }
Variant::~Variant() {}
Variant::Variant(bool b) : type_(Bool) { value_ =  static_cast<bool>(b); }
Variant::Variant(int i) : type_(Int)
{ value_ =  static_cast<int>(i); }
Variant::Variant(unsigned int i) : type_(UInt) { value_ =  static_cast<unsigned int>(i); }
Variant::Variant(double d) : type_(Double) { value_ = static_cast<double>(d); }
Variant::Variant(const std::string& string) : type_(String)
{ value_ =  string; }

Variant::Variant(const oskar::IntRange& r) : type_(IntRange) { value_ = r; }
Variant::Variant(const oskar::IntPositive& i) : type_(IntPositive), value_(i) {}
Variant::Variant(const oskar::IntRangeExt& r) : type_(IntRangeExt) { value_ = r; }
Variant::Variant(const oskar::RandomSeed& r) : type_(RandomSeed) { value_ = r; }
Variant::Variant(const oskar::DoubleRange& r) : type_(DoubleRange) { value_ = r; }
Variant::Variant(const oskar::DoubleRangeExt& r) : type_(DoubleRangeExt) { value_ = r; }
Variant::Variant(const oskar::IntList& l) : type_(IntList) { value_ = l; }
Variant::Variant(const oskar::DoubleList& l) : type_(DoubleList) { value_ = l; }
Variant::Variant(const oskar::StringList& l) : type_(StringList) { value_ = l; }
Variant::Variant(const oskar::OptionList& l) : type_(OptionList) { value_ = l; }
Variant::Variant(const oskar::DateTime& t) : type_(DateTime) { value_ = t; }
Variant::Variant(const oskar::Time& t) : type_(Time) { value_ = t; }
Variant::Variant(const oskar::InputFile& f) : type_(InputFile) { value_ = f; }
Variant::Variant(const oskar::InputFileList& l) : type_(InputFileList) { value_ = l; }
Variant::Variant(const oskar::InputDirectory& d) : type_(InputDirectory) { value_ = d; }
Variant::Variant(const oskar::OutputFile& f) : type_(OutputFile) { value_ = f; }

bool Variant::toBool(bool* ok) const { return convertToNumber_<bool>(ok); }
int Variant::toInt(bool* ok) const { return convertToNumber_<int>(ok); }
unsigned int Variant::toUInt(bool* ok) const {
    return convertToNumber_<unsigned int>(ok);
}
// Returns the variant as a double if the variant has type:
// Double, Int, Bool, StdString, or Str, or otherwise returns 0.0;
double Variant::toDouble(bool* ok) const { return convertToNumber_<double>(ok); }
std::string Variant::toString() const { return convertToString_(); }
const oskar::IntRange& Variant::toIntRange() const {
    //if (value_.is_singular()) init_(IntRange);
    return VAR::get<oskar::IntRange>(value_);
}
oskar::IntRange& Variant::toIntRange() {
    //if (value_.is_singular()) init_(IntRange);
    return VAR::get<oskar::IntRange>(value_);
}
const oskar::IntPositive& Variant::toIntPositive() const
{
    return VAR::get<oskar::IntPositive>(value_);
}
oskar::IntPositive& Variant::toIntPositive()
{
    return VAR::get<oskar::IntPositive>(value_);
}
const oskar::IntRangeExt& Variant::toIntRangeExt() const {
    return VAR::get<oskar::IntRangeExt>(value_);
}
oskar::IntRangeExt& Variant::toIntRangeExt() {
    return VAR::get<oskar::IntRangeExt>(value_);
}
const oskar::RandomSeed& Variant::toRandomSeed() const {
    return VAR::get<oskar::RandomSeed>(value_);
}
oskar::RandomSeed& Variant::toRandomSeed() {
    return VAR::get<oskar::RandomSeed>(value_);
}
const oskar::DoubleRange& Variant::toDoubleRange() const {
    return VAR::get<oskar::DoubleRange>(value_);
}
oskar::DoubleRange& Variant::toDoubleRange() {
    return VAR::get<oskar::DoubleRange>(value_);
}
const oskar::DoubleRangeExt& Variant::toDoubleRangeExt() const {
    return VAR::get<oskar::DoubleRangeExt>(value_);
}
oskar::DoubleRangeExt& Variant::toDoubleRangeExt() {
    return VAR::get<oskar::DoubleRangeExt>(value_);
}
const oskar::IntList& Variant::toIntList() const {
    return VAR::get<oskar::IntList>(value_);
}
oskar::IntList& Variant::toIntList() {
    return VAR::get<oskar::IntList>(value_);
}
const oskar::DoubleList& Variant::toDoubleList() const {
    return VAR::get<oskar::DoubleList>(value_);
}
oskar::DoubleList& Variant::toDoubleList() {
    return VAR::get<oskar::DoubleList>(value_);
}
const oskar::StringList& Variant::toStringList() const {
    return VAR::get<oskar::StringList>(value_);
}
oskar::StringList& Variant::toStringList() {
    /*if (value_.which() != metaTypeToVariantType_(type_)) init_(type_);*/
    if (value_.is_singular() || value_.which() != type_) init_(type_);
    return VAR::get<oskar::StringList>(value_);
}
const oskar::OptionList& Variant::toOptionList() const {
    return VAR::get<oskar::OptionList>(value_);
}
oskar::OptionList& Variant::toOptionList() {
    /*if (value_.which() != metaTypeToVariantType_(type_)) init_(type_);*/
    if (value_.is_singular() || value_.which() != type_) init_(type_);
    return VAR::get<oskar::OptionList>(value_);
}
const oskar::DateTime& Variant::toDateTime() const {
    return VAR::get<oskar::DateTime>(value_);
}
oskar::DateTime& Variant::toDateTime() {
    return VAR::get<oskar::DateTime>(value_);
}
const oskar::Time& Variant::toTime() const {
    return VAR::get<oskar::Time>(value_);
}
oskar::Time& Variant::toTime() {
    return VAR::get<oskar::Time>(value_);
}
const oskar::InputFile& Variant::toInputFile() const {
    return VAR::get<oskar::InputFile>(value_);
}
oskar::InputFile& Variant::toInputFile() {
    return VAR::get<oskar::InputFile>(value_);
}
const oskar::InputFileList& Variant::toInputFileList() const {
    return VAR::get<oskar::InputFileList>(value_);
}
oskar::InputFileList& Variant::toInputFileList() {
    return VAR::get<oskar::InputFileList>(value_);
}
const oskar::InputDirectory& Variant::toInputDirectory() const {
    return VAR::get<oskar::InputDirectory>(value_);
}
oskar::InputDirectory& Variant::toInputDirectory() {
    return VAR::get<oskar::InputDirectory>(value_);
}
const oskar::OutputFile& Variant::toOutputFile() const {
    return VAR::get<oskar::OutputFile>(value_);
}
oskar::OutputFile& Variant::toOutputFile() {
    return VAR::get<oskar::OutputFile>(value_);
}

void Variant::fromString(const std::string& s)
{
    using boost::lexical_cast;
    bool ok = false;
    switch (type_)
    {
    case Bool:
    {
        if (boost::iequals("true", s))
            value_ = true;
        else if (boost::iequals("false", s))
            value_ = false;
        break;
    }
    case Int:
    {
        try { value_ = lexical_cast<int>(s); }
        catch (boost::bad_lexical_cast&) {  }
        break;
    }
    case UInt:
    {
        try { value_ = lexical_cast<unsigned int>(s); }
        catch (boost::bad_lexical_cast&) {  }
        break;
    }
    case Double:
    {
        try { value_ = lexical_cast<double>(s); }
        catch (boost::bad_lexical_cast&) {  }
        break;
    }
    case String:
    {
        value_ = std::string(s);
        break;
    }
    case IntRange:
    case IntPositive:
    case IntRangeExt:
    {
        VAR::get<oskar::IntRange>(value_).fromString(s, &ok);
        break;
    }

    case DoubleRange:
    case DoubleRangeExt:
    {
        VAR::get<oskar::DoubleRange>(value_).fromString(s, &ok);
        break;
    }

    case IntList:
    {
        VAR::get<oskar::IntList>(value_).fromString(s, ',', &ok);
        break;
    }
    case StringList:
    {
        VAR::get<oskar::StringList>(value_).fromString(s, &ok);
        break;
    }
    case OptionList:
    {
        VAR::get<oskar::OptionList>(value_).fromString(s, &ok);
        break;
    }
    case DateTime:
    {
        VAR::get<oskar::DateTime>(value_).fromString(s, &ok);
        break;
    }
    case Time:
    {
        VAR::get<oskar::Time>(value_).fromString(s, &ok);
        break;
    }
    case InputFile:
    {
        VAR::get<oskar::InputFile>(value_).fromString(s, &ok);
        break;
    }
    case InputFileList:
    {
        VAR::get<oskar::InputFileList>(value_).fromString(s, &ok);
        break;
    }
    case InputDirectory:
    {
        VAR::get<oskar::InputDirectory>(value_).fromString(s, &ok);
        break;
    }
    case OutputFile:
    {
        VAR::get<oskar::OutputFile>(value_).fromString(s);
        break;
    }
    default:
        break;
    };
}

// Method that does not change the type of the current variant.
void Variant::fromVariant(const Variant& other, bool* ok)
{
    if (ok) *ok = true;

    // Type of this is already the same as other.
    if (type_ == other.type_)
    {
//        Variant temp(other);
//        std::swap(value_, temp.value_);
        *this = other;
    }
    // Type of other is different from type of this.
    // Make sure type of this dons't change by casting
    // other to this type when assigning new value.
    // FIXME complex types here probably don't work or throw!
    // TODO set 'ok' argument appropriately in all cases.
    else
    {
        switch (type_)
        {
            case Variant::Bool: {
                value_ = other.toBool(ok);
                break;
            }
            case Variant::Int: {
                value_ = other.toInt(ok);
                break;
            }
            case Variant::UInt: {
                value_ = other.toUInt(ok);
                break;
            }
            case Variant::Double: {
                value_ = other.toDouble(ok);
                break;
            }
            case Variant::String: {
                value_ = other.toString();
                break;
            }
            // --------------------------------------------
            case Variant::IntRange: {
                value_ = other.toIntRange();
                break;
            }
            case Variant::IntPositive: {
                value_ = other.toIntPositive();
                break;
            }
            case Variant::IntRangeExt: {
                value_ = other.toIntRangeExt();
                break;
            }
            case Variant::DoubleRange: {
                value_ = other.toDoubleRange();
                break;
            }
            case Variant::DoubleRangeExt: {
                value_ = other.toDoubleRangeExt();
                break;
            }
            // --------------------------------------------
            case Variant::IntList: {
                value_ = other.toIntList();
                break;
            }
            case Variant::DoubleList: {
                value_ = other.toDoubleList();
                break;
            }
            case Variant::StringList: {
                value_ = other.toStringList();
                break;
            }
            case Variant::OptionList: { // TODO i've no idea if this is valid?!
                value_ = other.toOptionList();
                break;
            }
            // --------------------------------------------
            case Variant::DateTime: {
                value_ = other.toDateTime();
                break;
            }
            case Variant::Time: {
                value_ = other.toTime();
                break;
            }
            case Variant::InputFile: {
                value_ = other.toInputFile();
                break;
            }
            case Variant::InputFileList: {
                value_ = other.toInputFileList();
                break;
            }
            case Variant::InputDirectory: {
                value_ = other.toInputDirectory();
                break;
            }
            case Variant::OutputFile: {
                value_ = other.toOutputFile();
                break;
            }
            default:
                if (ok) *ok = false;
                value_ = privateVariant();
                break;
        };
    }
}

Variant::MetaTypeId Variant::type() const
{
//    cout << string(80, '-') << endl;
//    cout << __PRETTY_FUNCTION__ << " [Line:" << __LINE__ << "]" << endl;
//    cout << "  TypeName       = " << typeName() << endl;
//    cout << "  is valid?      = " << (value_.is_singular() ? "false" : "true") << endl;
    if (!value_.is_singular()) assert(value_.which() == type_);
//    cout << "  type_          = " << (int)type_ << endl;
//    cout << string(80, '-') << endl;
    return type_;
}

bool Variant::isSet() const
{
    //cout << __PRETTY_FUNCTION__ << " type_ = " << type_ <<endl;
    if (value_.is_singular()) {
//        cout << "is singular..." << endl;
        return false;
    }
    //cout << __PRETTY_FUNCTION__ << endl;

    // FIXME: some of the more complex types require their own is set function
    // to be called rather than just checking if the type exists.
    // ie. the variant may have a type but the type might not be 'set'
    switch (type_)
    {
        case IntRangeExt:
            // TODO check against type in the variant here rather than just if set.
            if (!value_.is_singular()) return toIntRangeExt().isSet();
            break;
        case DoubleRangeExt:
            // TODO check against type in the variant here rather than just if set.
            if (!value_.is_singular()) return toDoubleRangeExt().isSet();
            break;
        case OptionList:
            if (!value_.is_singular()) return toOptionList().isSet();
            break;
        default:
            break;
    };
    //cout << __PRETTY_FUNCTION__ << endl;
    return !this->toString().empty();
}

bool Variant::isValid() const
{
    return type_ != Invalid;
}

std::string Variant::convertToString_() const
{
    using boost::lexical_cast;
    if (value_.is_singular()) return std::string();

    // Switch on the (meta) type we are converting **from**.
    switch (type_)
    {
        case Invalid:
            return std::string();

        case Bool:
        {
            bool b = VAR::get<bool>(value_);
            if (b) return std::string("true");
            else return std::string("false");
        }
        case Int:
            return lexical_cast<std::string>(VAR::get<int>(value_));
        case UInt:
            return lexical_cast<std::string>(VAR::get<unsigned int>(value_));
        case Double:
        {
            std::ostringstream ss;
            ss << VAR::get<double>(value_);
            return ss.str();
        }
        case String: {
            try {
                return VAR::get<std::string>(value_);
            }
            catch (const ttl::exception& e)
            {
//                std::cout << typeName() << std::endl;
//                std::cout << e.what() << std::endl;
                return std::string();
            }
            break;
        }

        case IntRange:
            return VAR::get<oskar::IntRange>(value_).toString();
        case IntPositive:
            return VAR::get<oskar::IntPositive>(value_).toString();
        case IntRangeExt:
            return VAR::get<oskar::IntRangeExt>(value_).toString();

        case DoubleRange:
            return VAR::get<oskar::DoubleRange>(value_).toString();
        case DoubleRangeExt:
            return VAR::get<oskar::DoubleRangeExt>(value_).toString();

        case IntList:
            return VAR::get<oskar::IntList>(value_).toString();
        case DoubleList:
            return std::string(VAR::get<oskar::DoubleList>(value_).toStr());
        case StringList:
            return VAR::get<oskar::StringList>(value_).toString();
        case OptionList:
            return this->toOptionList().toString();

        case DateTime:
            return VAR::get<oskar::DateTime>(value_).toString();
        case Time:
            return VAR::get<oskar::Time>(value_).toString();

//        case InputFile:
//            return get<oskar::InputFile>(value_).toString();
        case InputFileList:
            return VAR::get<oskar::InputFileList>(value_).toString();
        case InputDirectory:
            return VAR::get<oskar::InputDirectory>(value_).toString();
        case OutputFile:
            return VAR::get<oskar::OutputFile>(value_).toString();

        default:
            break;
    };

    return std::string();
}

void Variant::init_(Variant::MetaTypeId t)
{
//    cout << __PRETTY_FUNCTION__ << endl;
    switch (t)
    {
        case Bool:
        {
            value_ = false;
            break;
        }
        case Int:
        {
            value_ = 0;
            break;
        }
        case UInt:
        {
            value_ = 0u;
            break;
        }
        case Double:
        {
            value_ = (double)0;
            break;
        }
        case String:
        {
            value_ = std::string("");
            break;
        }
        case IntRange:
        {
            value_ = oskar::IntRange();
            break;
        }
        case IntPositive:
        {
            value_ = static_cast<oskar::IntRange>(oskar::IntPositive());
            break;
        }
        case IntRangeExt:
        {
            value_ = static_cast<oskar::IntRange>(oskar::IntRangeExt());
            break;
        }
        case DoubleRange:
        {
            value_ = oskar::DoubleRange();
            break;
        }
        case DoubleRangeExt:
        {
            value_ = static_cast<oskar::DoubleRange>(oskar::DoubleRangeExt());
            break;
        }
        case IntList:
        {
            value_ = oskar::IntList();
            break;
        }
        case DoubleList:
        {
            value_ = oskar::DoubleList();
            break;
        }
        case StringList:
        {
            value_ = oskar::StringList();
            break;
        }
        case OptionList:
        {
            // NOTE: this constructor will generate a useless OptionList!
            value_ = oskar::OptionList();
            break;
        }
        case DateTime:
        {
            value_ = oskar::DateTime();
            break;
        }
        case Time:
        {
            value_ = oskar::Time();
            break;
        }

//        case InputFile:
//        {
//            value_ = oskar::InputFile();
//            break;
//        }

        case InputFileList:
        {
            value_ = oskar::InputFileList();
            break;
        }

        case InputDirectory:
        {
            value_ = oskar::InputDirectory();
            break;
        }
        case OutputFile:
        {
            value_ = oskar::OutputFile();
            break;
        }

        default:
        {
            break;
        }
    };
}

// ok == true indicates conversion was possible.
template <typename T> T Variant::convertToNumber_(bool* ok) const
{
    using boost::lexical_cast;

    // If the boost variant isn't set isn't set, return.
    if (value_.is_singular()) {
        if (ok) *ok = false;
        return 0;
    }

    // Get the meta type we are trying to convert to.
    MetaTypeId newType = getType_<T>();
    MetaTypeId currentType = type_;

#if 0
    cout << std::string(80, '-') << endl;
    cout << __PRETTY_FUNCTION__ << "  [Line:" << __LINE__ << "]" << endl;
    cout << "  TO   : " << typeName_(newType) << endl;
    cout << "  FROM : " << typeName_(currentType) << endl;
    cout << std::string(80, '-') << endl;
#endif

    if (ok) *ok = true;

    if (currentType == Invalid) {
        if (ok) *ok = false;
    }

    if (newType == Invalid) {
        if (ok) *ok = false;
        return 0;
    }

    // No conversion necessary.
    if (newType == currentType) {
        return VAR::get<T>(value_);
    }

    // Switch on the type we are converting to, followed by the type from.
    switch (newType)
    {
        case Bool:
        {
            switch (currentType)
            {
                case Int: // int value, return a bool (T == bool)
                    return static_cast<T>(VAR::get<int>(value_));
                case UInt:
                    return static_cast<T>(VAR::get<unsigned int>(value_));
                case RandomSeed:  // same as IntRange
                    return static_cast<T>(VAR::get<oskar::RandomSeed>(value_).getInt());
                case IntRangeExt: // same as IntRange
                    return static_cast<T>(VAR::get<oskar::IntRangeExt>(value_).getInt());
                case IntRange:
                    return static_cast<T>(VAR::get<oskar::IntRange>(value_).getInt());
                case IntPositive:
                    return true;
                case Double: // double value, return bool.
                    return VAR::get<double>(value_) != 0.0;
                case DoubleRange:
                    return VAR::get<oskar::DoubleRange>(value_).getDouble() != 0.0;
                case DoubleRangeExt:
                    return VAR::get<oskar::DoubleRangeExt>(value_).getDouble() != 0.0;
                case String:
                    return (VAR::get<std::string>(value_).size() > 0);
                default:
                    break;
            };
        }
        break;

        // ==== Conversions TO int ====
        case Int:
        {
            switch (currentType)
            {
                case Double: // double value, return an int
                    return static_cast<T>(VAR::get<double>(value_));
                case DoubleRange:
                    return static_cast<T>(VAR::get<oskar::DoubleRange>(value_).getDouble());
                case DoubleRangeExt:
                    return static_cast<T>(VAR::get<oskar::DoubleRangeExt>(value_).getDouble());
                case UInt:
                    return static_cast<T>(VAR::get<unsigned int>(value_));
                case IntRange:
                    return VAR::get<oskar::IntRange>(value_).getInt();
                case RandomSeed:
                    return VAR::get<oskar::RandomSeed>(value_).getInt();
                case IntRangeExt:
                    return VAR::get<oskar::IntRangeExt>(value_).getInt();
                case IntPositive:
                    return VAR::get<oskar::IntPositive>(value_).getInt();
                case String: // std::string value, return an int.
                {
                    try {
                        return lexical_cast<int>(VAR::get<std::string>(value_));
                    }
                    catch (boost::bad_lexical_cast&) {
                        if (ok) { *ok = false; }
                        return 0;
                    }
                    //return boost::lexical_cast<int>(boost::get<std::string>(value_));
                }
                default:
                    break;
            };
        }
        break;

        // ==== Conversions TO unsigned int ====
        case UInt:
        {
            switch (currentType)
            {
                case Int: {
                    int i = VAR::get<int>(value_);
                    if (i < 0) { if (ok) { *ok = false; } }
                    return static_cast<T>(i);
                }

                case Double: {
                    double d = VAR::get<double>(value_);
                    if (d < 0) { if (ok) { *ok = false; } }
                    return static_cast<T>(d);
                }

                case DoubleRange:
                case DoubleRangeExt:
                {
                    double d = VAR::get<oskar::DoubleRange>(value_).getDouble();
                    if (d < 0) { if (ok) { *ok = false; } }
                    return static_cast<T>(d);
                }

                case RandomSeed:
                case IntRange:
                case IntPositive:
                case IntRangeExt:
                {
                    int i = VAR::get<oskar::IntRange>(value_).getInt();
                    if (i < 0) { if (ok) { *ok = false; } }
                    return static_cast<T>(i);
                }

                case String: // std::string value, return an int.
                {
                    try {
                        return lexical_cast<unsigned int>(VAR::get<std::string>(value_));
                    }
                    catch (boost::bad_lexical_cast&) {
                        if (ok) { *ok = false; }
                        return 0;
                    }
                }
                default:
                    break;
            };
        }
        break;

        // ==== Conversions TO double ====
        case Double:
        {
            switch (currentType) {

                case DoubleRange:
                    return VAR::get<oskar::DoubleRange>(value_).getDouble();
                case DoubleRangeExt: {
                    return VAR::get<oskar::DoubleRangeExt>(value_).getDouble();
                }
                case Int: //  int value, return a double
                    return static_cast<T>(VAR::get<int>(value_));
                case UInt: // unsigned int value, return a double
                    return static_cast<T>(VAR::get<unsigned int>(value_));
                case RandomSeed:
                    return static_cast<T>(VAR::get<oskar::RandomSeed>(value_).getInt());
                case IntRange:
                    return static_cast<T>(VAR::get<oskar::IntRange>(value_).getInt());
                case IntPositive:
                    return static_cast<T>(VAR::get<oskar::IntPositive>(value_).getInt());
                case IntRangeExt:
                    return static_cast<T>(VAR::get<oskar::IntRangeExt>(value_).getInt());
                case String:
                    try {
                        return lexical_cast<double>(VAR::get<std::string>(value_));
                    }
                    catch (boost::bad_lexical_cast&) {
                        if (ok) { *ok = false; }
                        return 0;
                    }
                default:
                    break;
            };
        }
        break;

        default:
            break;
    };

    return 0;
}

template <typename T> Variant::MetaTypeId Variant::getType_() const
{
    // FIXME this wont work without a 1 to 1 mapping of variant types to meta types.
    // TODO some form of indirection.
    privateVariant v((T)0);
    return static_cast<MetaTypeId>(v.which());
}

const char* Variant::typeName_(Variant::MetaTypeId type) const
{
    switch (type)
    {
        case Invalid: return "invalid";

        case Bool:   return "bool";
        case Int:    return "int";
        case UInt:   return "unsigned int";
        case Double: return "double";
        case String: return "std::string";

        case IntRange:       return "oskar::IntRange";
        case IntPositive:    return "oskar::IntPositive";
        case IntRangeExt:    return "oskar::IntRangeExt";
        case RandomSeed:     return "oskar::RandomSeed";
        case DoubleRange:    return "oskar::DoubleRange";
        case DoubleRangeExt: return "oskar::DoubleRangeExt";

        case IntList:    return "oskar::IntList";
        case DoubleList: return "oskar::DoubleList";
        case StringList: return "oskar::StringList";
        case OptionList: return "oskar::OptionList";

        case DateTime:   return "oskar::DateTime";
        case Time:       return "oskar::Time";

        case InputFile:      return "oskar::InputFile";
        case InputFileList:  return "oskar::InputFileList";
        case InputDirectory: return "oskar::InputDirectory";
        case OutputFile:     return "oskar::OutputFile";

        default: return "ERROR: Unknown type";
    };
    return "invalid";
}

int Variant::metaTypeToVariantType_(const MetaTypeId& type) const
{
    switch (type)
    {
        case Invalid: return -999;

        case Bool:   return 0;
        case Int:    return 1;
        case UInt:   return 2;
        case Double: return 3;

        case String: return 4;

        case IntRange: return 5;
        case IntPositive: return 6;
        case IntRangeExt: return 7;
        case RandomSeed: return 8;

        case DoubleRange: return 9;
        case DoubleRangeExt: return 10;

        case IntList:     return 11;
        case DoubleList:  return 12;
        case StringList:  return 13;
        case OptionList:  return 14;

        case DateTime:        return 15;
        case Time:            return 16;

        case InputFile:       return 17;
        case InputFileList:   return 18;
        case InputDirectory:  return 19;
        case OutputFile:      return 20;
    };
    return 0;
}

bool Variant::isValueEqual(const Variant& other) const
{
    switch (other.type())
    {
        case Bool:
        {
            bool other_ = other.toBool();
            bool this_ = this->toBool();
            return (this_ == other_);
        }
        case IntRange:    // same as Int.
        case IntPositive: // same as Int
        case IntRangeExt: // same as Int ??!
        case RandomSeed:  // Same as int in all cases?
        case Int:
        {
            bool ok = false;
            int other_ = other.toInt(&ok);
            if (!ok) return false;
            int this_  = this->toInt(&ok);
            if (!ok) return false;
            return (this_ == other_);
        }
        case UInt:
        {
            bool ok = false;
            unsigned int other_ = other.toUInt(&ok);
            if (!ok) return false;
            unsigned int this_  = this->toUInt(&ok);
            if (!ok) return false;
            return (this_ == other_);
        }
        case DoubleRangeExt:
        case DoubleRange:
        case Double:
        {
            bool ok = false;
            double other_ = other.toDouble(&ok);
            if (!ok) return false;
            double this_  = this->toDouble(&ok);
            if (!ok) return false;
            return (this_ == other_);
        }
        case String:
        {
            std::string other_ = other.toString();
            std::string this_ = this->toString();
            return (this_ == other_);
        }
        case IntList:
            return this->toIntList().isEqual(other.toIntList());
        case DoubleList:
            return this->toDoubleList().isEqual(other.toDoubleList());
        case StringList:
            return this->toStringList().isEqual(other.toStringList());
        case OptionList:
            return (this->toOptionList().value() == other.toOptionList().value());
        case DateTime:
            return this->toDateTime().isEqual(other.toDateTime());
        case Time:
            return this->toTime().isEqual(other.toTime());
//        case InputFile:
//            return (this->toInputFile() == other.toInputFile());
        case InputFileList:
            return (this->toInputFileList() == other.toInputFileList());
        case InputDirectory:
            return (this->toInputDirectory() == other.toInputDirectory());
        case OutputFile:
            return (this->toOutputFile() == other.toOutputFile());
        case Invalid:
        default:
            break;
    };
    return false;
}

bool Variant::operator==(const Variant& other) const
{ return isValueEqual(other); }

bool Variant::operator!=(const Variant& other) const
{ return !isValueEqual(other); }

#if 0
Variant& Variant::operator=(const Variant& other)
{
    //cout << __PRETTY_FUNCTION__ << endl;
    // copy, swap http://www.cplusplus.com/articles/y8hv0pDG/
    // probably not needed here...?

    Variant tmp(other);
    // Assignment wants to make sure the type doesn't change.
    // So needs to convert from the variant type of other to the variant type
    // of this.

    // If this variant is not currently set or the type isn't going to be
    // changed, swap attributes with tmp.
    if (type_ == Invalid || tmp.type_ == type_) {
//        cout << "HERE A" << endl;
//        cout << value_.which() << " " << (int)type_ << endl;
        if (value_.which() != 0) {
            assert(value_.which() == (int)type_);
            assert(tmp.value_.which() == (int)type_);
        }
        std::swap(type_, tmp.type_);
        std::swap(value_, tmp.value_);
    }
    // If the types are different convert value of tmp to this type.
    else {
//        cout << "HERE B" << endl;
//        cout << value_.which() << " " << (int)type_ << endl;
        if (value_.which() != 0)
            assert(value_.which() == (int)type_);
        switch (type_)
        {
            case Bool: {
                value_ = tmp.toBool();
                break;
            }
            case Int: {
                value_ = tmp.toInt();
                break;
            }
            case UInt: {
                value_ = tmp.toUInt();
                break;
            }
            case Double: {
                value_ = tmp.toDouble();
                break;
            }
            case String: {
                value_ = tmp.toString();
                break;
            }
            // --------------------------
            case IntRange:
            case IntPositive:
            case IntRangeExt:
            case DoubleRange:
            case DoubleRangeExt:

            case IntList:
            case IntListExt:
            case DoubleList:
            case StringList:

            case DateTime:
            case Time:

            default: {
                value_ = privateVariant();
                break;
            }
        };
    }

    return *this;
}
#endif

} // namespace oskar
