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

#ifndef OSKAR_SETTINGS_VARIANT_HPP_
#define OSKAR_SETTINGS_VARIANT_HPP_

#include <ttl/var/variant.hpp>
#include <string>
#include <memory>
#include <oskar_settings_types.hpp>

#ifdef __GNUC__
#define DEPRECATED(func) func __attribute__ ((deprecated))
#elif defined(_MSC_VER)
#define DEPRECATED(func) __declspec(deprecated) func
#else
#pragma message("WARNING: You need to implement DEPRECATED for this compiler")
#define DEPRECATED(func) func
#endif

namespace oskar {

/*
 * Union like container for a variety of types. This is motivated by the
 * need for a container that is capable of manipulating types in a uniform
 * manner and not limited, in the case of the C/C++ union, to primitive data
 * types.
 */
class Variant
{
public:
    /* NOTE this list *HAS* to match the order of types in the variant */
    enum MetaTypeId
    {
        Invalid = -1,
        // Fundamental types
        Bool = 0,
        Int,
        UInt,
        Double,
        String,
        // Derived types
        DateTime,
        DoubleList,
        DoubleRange,
        DoubleRangeExt,
        InputDirectory,
        InputFile,
        InputFileList,
        IntList,
        IntPositive,
        IntRange,
        IntRangeExt,
        OptionList,
        OutputFile,
        RandomSeed,
        StringList,
        Time
    };

private:
        ttl::var::variant<
                bool,
                int,
                unsigned int,
                double,
                std::string,
                oskar::DateTime,
                oskar::DoubleList,
                oskar::DoubleRange,
                oskar::DoubleRangeExt,
                oskar::InputDirectory,
                oskar::InputFile,
                oskar::InputFileList,
                oskar::IntList,
                oskar::IntPositive,
                oskar::IntRange,
                oskar::IntRangeExt,
                oskar::OptionList,
                oskar::OutputFile,
                oskar::RandomSeed,
                oskar::StringList,
                oskar::Time
                > value_;
        MetaTypeId type_;
public:
    // === Default constructor & Destructor. ==================================
    inline Variant() : type_(Invalid) {}
    Variant(const Variant& other);
    Variant(MetaTypeId type);
    Variant(MetaTypeId type, const std::string& s);
    ~Variant();

    // === Constructors with type argument. ===================================
    Variant(bool b);
    Variant(int i);
    Variant(unsigned int i);
    Variant(double d);
    Variant(const std::string& string);
    Variant(const oskar::IntRange& r);
    Variant(const oskar::IntPositive& i);
    Variant(const oskar::IntRangeExt& i);
    Variant(const oskar::RandomSeed& r);
    Variant(const oskar::DoubleRange& d);
    Variant(const oskar::DoubleRangeExt& d);
    Variant(const oskar::IntList& l);
    Variant(const oskar::DoubleList& l);
    Variant(const oskar::StringList& l);
    Variant(const oskar::OptionList& l);
    Variant(const oskar::DateTime& t);
    Variant(const oskar::Time& t);
    Variant(const oskar::InputFile& f);
    Variant(const oskar::InputFileList& l);
    Variant(const oskar::InputDirectory& d);
    Variant(const oskar::OutputFile& f);

    static MetaTypeId typeFromString(const std::string& s)
    {
        MetaTypeId t = Invalid;

        std::string sType(s);
        // Convert to upper case
        //std::transform(sType.begin(), sType.end(), sType.begin(), ::toupper);

        if (sType == "Bool")
            t = Bool;
        else if (sType == "Int")
            t = Int;
        else if (sType == "UInt")
            t = UInt;
        else if (sType == "Double")
            t = Double;
        else if (sType == "String")
            t = String;
        else if (sType == "IntRange")
            t = IntRange;
        else if (sType == "IntPositive")
            t = IntPositive;
        else if (sType == "IntRangeExt")
            t = IntRangeExt;
        else if (sType == "RandomSeed")
            t = RandomSeed;
        else if (sType == "DoubleRange")
            t = DoubleRange;
        else if (sType == "DoubleRangeExt")
            t = DoubleRangeExt;
        else if (sType == "IntList")
            t = IntList;
        else if (sType == "DoubleList")
            t = DoubleList;
        else if (sType == "StringList")
            t = StringList;
        else if (sType == "OptionList")
            t = OptionList;
        else if (sType == "DateTime")
            t = DateTime;
        else if (sType == "Time")
            t = Time;
        else if (sType == "InputFileList")
            t = InputFileList;
        else if (sType == "InputDirectory")
            t = InputDirectory;
        else if (sType == "OutputFile")
            t = OutputFile;

        return t;
    }

    // === General utility accessors. =========================================
    Variant::MetaTypeId type() const;
    const char* typeName() const { return typeName_(type_); }
    bool isSet() const;
    bool isValid() const;

    // === Accessors (with conversion, where applicable) to specified types. ==

    // Returns the variant as a bool if the variant has type() Bool.
    // Returns true if the variant has type() Int or Double and the value is
    // non-zero.
    // Returns true if the variant has type Str or StdString and is not-empty.
    // Otherwise returns false.
    bool toBool(bool* ok = 0) const;
    int toInt(bool* ok = 0) const;
    unsigned int toUInt(bool* ok = 0) const;
    // Returns the variant as a double if the variant has type:
    // Double, Int, Bool, StdString, or Str, or otherwise
    // returns 0.0;
    double toDouble(bool* ok = 0) const;
    // Returns a copy in order to be able to convert?
    std::string toString() const;
    const oskar::IntRange& toIntRange() const;
    oskar::IntRange& toIntRange();
    const oskar::IntPositive& toIntPositive() const;
    oskar::IntPositive& toIntPositive();
    const oskar::IntRangeExt& toIntRangeExt() const;
    const oskar::RandomSeed& toRandomSeed() const;
    oskar::RandomSeed& toRandomSeed();
    oskar::IntRangeExt& toIntRangeExt();
    const oskar::DoubleRange& toDoubleRange() const;
    oskar::DoubleRange& toDoubleRange();
    const oskar::DoubleRangeExt& toDoubleRangeExt() const;
    oskar::DoubleRangeExt& toDoubleRangeExt();
    const oskar::IntList& toIntList() const;
    oskar::IntList& toIntList();
    const oskar::DoubleList& toDoubleList() const;
    oskar::DoubleList& toDoubleList();
    const oskar::StringList& toStringList() const;
    oskar::StringList& toStringList();
    const oskar::OptionList& toOptionList() const;
    oskar::OptionList& toOptionList();
    const oskar::DateTime& toDateTime() const;
    oskar::DateTime& toDateTime();
    const oskar::Time& toTime() const;
    oskar::Time& toTime();
    const oskar::InputFile& toInputFile() const;
    oskar::InputFile& toInputFile();
    const oskar::InputFileList& toInputFileList() const;
    oskar::InputFileList& toInputFileList();
    const oskar::InputDirectory& toInputDirectory() const;
    oskar::InputDirectory& toInputDirectory();
    const oskar::OutputFile& toOutputFile() const;
    oskar::OutputFile& toOutputFile();

    void fromString(const std::string& s);
    void fromVariant(const Variant& other, bool* ok = 0);

    bool isValueEqual(const Variant& other) const;
    bool operator==(const Variant& other) const;
    bool operator!=(const Variant& other) const;

#if 0
    Variant& operator=(const Variant& other);
#endif

private:
    // TODO init_ is deprecated in favour of not settings the variant....
    void init_(Variant::MetaTypeId t);
    // Template function so has to be in the header?
    template <typename T> T convertToNumber_(bool* ok = 0) const;
    std::string convertToString_() const;
    // Template function so has to be in the header?
    template <typename T> MetaTypeId getType_() const;
    const char* typeName_(MetaTypeId type) const;
    DEPRECATED(int metaTypeToVariantType_(const MetaTypeId& type) const);
};


} // namespace oskar
#endif /* OSKAR_SETTINGS_VARIANT_HPP_ */
