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

#include <gtest/gtest.h>

#include <string>
#include <vector>
#include <iostream>

#include "../oskar_SettingsVariant.hpp"

using namespace oskar;
using namespace std;

/*
 * => Decide on behavior of variants constructed with a type only.
 *    -> Should these be valid or not?
 *    -> Should there be a way of finding out if they have been set?
 *
 * Checks:
 * - Constructor(s)
 * - Type (ID, name)
 * - toString()
 * - fromString()
 * - Conversion functions
 * - Equality operator/function
 */

TEST(Variant, Invalid)
{
    Variant v;

    ASSERT_EQ(Variant::Invalid, v.type());
    ASSERT_STREQ("invalid", v.typeName());

    ASSERT_STREQ("", v.toString().c_str());
    v.fromString(string());
    ASSERT_STREQ("", v.toString().c_str());
    ASSERT_EQ(Variant::Invalid, v.type());

    bool ok = false;
    ASSERT_FALSE(v.toBool());
    ASSERT_EQ(0, v.toInt(&ok));
    ASSERT_FALSE(ok); // false as we are converting from an invalid value!
    ASSERT_DOUBLE_EQ(0.0, v.toDouble(&ok));
    ASSERT_FALSE(ok);
}

TEST(Variant, Bool)
{
    // Use case: Declare from type enum constructor and use.
    {
        bool ok = false;
        Variant v(Variant::Bool);
        ASSERT_FALSE(v.isSet());
        // TODO Not been set so fail casting to bool somehow...?
        ASSERT_FALSE(v.toBool(&ok));
        ASSERT_FALSE(ok);
        v.fromString("true");
        ASSERT_TRUE(v.toBool());
        v = false;
        ASSERT_FALSE(v.toBool());
        ASSERT_STREQ("false", v.toString().c_str());
    }

    // Use case: Construct from string as specified type.
    {
        Variant v = Variant(Variant::Bool, "true");
        ASSERT_EQ(true, v.toBool());
    }

    {
        Variant v(true);
        ASSERT_EQ(Variant::Bool, v.type());
        ASSERT_EQ(true, v.toBool());
    }

    // Declare from assignment operator
    {
        Variant v = true;
        ASSERT_EQ(Variant::Bool, v.type());
        ASSERT_STREQ("bool", v.typeName());
        ASSERT_EQ(true, v.toBool());
    }
}

TEST(Variant, Int)
{
    {
        int value = 2;
        Variant v(value);
        ASSERT_EQ(Variant::Int, v.type());
        ASSERT_EQ(value, v.toInt());

        // Conversion functions.
        ASSERT_TRUE(v.toBool());
        ASSERT_DOUBLE_EQ((double)value, v.toDouble());
        ASSERT_STREQ("2", v.toString().c_str());
        v = 0;
        ASSERT_FALSE(v.toBool());

        // Assignment operator
        Variant other = v;
        ASSERT_EQ(other.type(), v.type());

        // Copy constructor
        Variant copy(v);
        ASSERT_EQ(copy.type(), v.type());
    }

    {
        Variant v(Variant::Int);
        v.fromString("5");
        ASSERT_EQ(5, v.toInt());
        ASSERT_STREQ("5", v.toString().c_str());
    }
}

TEST(Variant, UInt)
{
    Variant v((unsigned int)3);
    ASSERT_EQ(Variant::UInt, v.type());
    ASSERT_EQ((unsigned int)3, v.toUInt());

    // Conversion functions.
    ASSERT_TRUE(v.toBool());
    bool ok = false;
    ASSERT_EQ(3, v.toInt(&ok));
    ASSERT_TRUE(ok);
    ASSERT_EQ(3.0, v.toDouble(&ok));
    ASSERT_TRUE(ok);
    ASSERT_STREQ("3", v.toString().c_str());

    // Assignment operator.
    Variant other = v;
    ASSERT_EQ(other.type(), v.type());

    // Copy constructor.
    Variant copy(v);
    ASSERT_EQ(copy.type(), v.type());
}

TEST(Variant, Double)
{
    Variant v;
    double dblValue = 1.2345;
    v = dblValue;
    ASSERT_EQ(Variant::Double, v.type());
    ASSERT_DOUBLE_EQ(dblValue, v.toDouble());

    // Conversion functions.
    ASSERT_TRUE(v.toBool());
    bool ok = false;
    ASSERT_EQ(1, v.toInt(&ok));
    ASSERT_TRUE(ok);
    EXPECT_STREQ("1.2345", v.toString().c_str());

    // Assignment operator
    Variant other = v;
    ASSERT_EQ(other.type(), v.type());

    // Copy constructor
    Variant copy(v);
    ASSERT_EQ(copy.type(), v.type());
}

TEST(Variant, IntRange)
{
    {
        const Variant v(Variant::IntRange);
        ASSERT_FALSE(v.isSet());
        ASSERT_EQ(Variant::IntRange, v.type());
        // FIXME This will fail as the variant isn't initialised ...
        // and cant be as the variant is const.
        // If the correct behaviour is not to throw but return but return a
        // default constructed IntRange, then the Variant will have to be
        // initialised on construction in this case.
        const IntRange& r = v.toIntRange();
    }

    // Range used as a positive int.
    {
        Variant v(IntRange(1, INT_MAX, 2));
        ASSERT_EQ(Variant::IntRange, v.type());
        ASSERT_STREQ("oskar::IntRange", v.typeName());
        bool ok = false;
        ASSERT_EQ(2, v.toInt(&ok));
        ASSERT_TRUE(ok);
        ASSERT_EQ((unsigned int)2, v.toUInt(&ok));

        ASSERT_EQ(2.0, v.toDouble(&ok));
        ASSERT_TRUE(ok);

        ASSERT_TRUE(v.toBool());

        ASSERT_STREQ("2", v.toString().c_str());
    }
    // Invalid use.
    {
        Variant v(IntRange(0, 5, 3));
        bool ok = false;
        ASSERT_EQ(3, v.toInt(&ok));
        ASSERT_TRUE(ok);

        ASSERT_EQ(0, v.toIntRange().min());
        ASSERT_EQ(5, v.toIntRange().max());
        ASSERT_EQ(3, v.toIntRange().getInt());

        v.toIntRange().set(-1, &ok);
        ASSERT_EQ(0, v.toIntRange().getInt());
        ASSERT_FALSE(ok);

        v.toIntRange().set(7, &ok);
        ASSERT_EQ(5, v.toIntRange().getInt());
        ASSERT_FALSE(ok);

        v.toIntRange().set(4, &ok);
        ASSERT_EQ(4, v.toIntRange().getInt());
        ASSERT_TRUE(ok);
    }
}

TEST(Variant, RandomSeed)
{
    using oskar::RandomSeed;
    using oskar::IntRange;
    {
//        Variant v(oskar::RandomSeed);
        Variant v(RandomSeed());
//        Variant v(oskar::IntRange);
//        v.type();
        //    ASSERT_STREQ("oskar::RandomSeed", v.typeName());
    }
    {
//        Variant v(RandomSeed());
//        v.type();
    }
}

TEST(Variant, IntPositive)
{
    using oskar::IntPositive;
    Variant v(IntPositive(10));
    ASSERT_EQ(Variant::IntPositive, v.type());
    ASSERT_STREQ("oskar::IntPositive", v.typeName());

    bool ok = false;
    ASSERT_EQ(10, v.toInt(&ok));
    ASSERT_TRUE(ok);

    IntPositive i = v.toIntPositive();
    ASSERT_EQ(10, i.getInt(&ok));
    ASSERT_TRUE(ok);

    v = IntPositive(-1);
    ASSERT_EQ(1, v.toInt(&ok));
}

TEST(Variant, IntRangeExt)
{
    bool ok = false;
    int min = 0;
    int max = INT_MAX;
    string smin = "time";

    Variant v = IntRangeExt(min, max, smin);
    ASSERT_EQ(Variant::IntRangeExt, v.type());
    ASSERT_STREQ("oskar::IntRangeExt", v.typeName());

    ASSERT_FALSE(v.isSet());
    ASSERT_EQ(0, v.toInt(&ok));
    ASSERT_STREQ("", v.toString().c_str());

    ASSERT_TRUE(ok);
    v.toIntRangeExt().fromString(smin, &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ(min-1, v.toIntRangeExt().getInt(&ok));
    ASSERT_FALSE(ok);

    ASSERT_STREQ(smin.c_str(), v.toIntRangeExt().toString(&ok).c_str());
    ASSERT_TRUE(ok);
}

TEST(Variant, DoubleRange)
{
    using oskar::DoubleRange;
    Variant v = DoubleRange(0.0, 10.0, 1.0);
    ASSERT_EQ(Variant::DoubleRange, v.type());
    ASSERT_STREQ("oskar::DoubleRange", v.typeName());

    bool ok = false;
    ASSERT_DOUBLE_EQ(1.0, v.toDouble(&ok));
    ASSERT_TRUE(ok);
}

TEST(Variant, DoubleRangeExt)
{
    using oskar::DoubleRangeExt;
    Variant v = DoubleRangeExt(0.0, 180.0, "all");
    ASSERT_EQ(Variant::DoubleRangeExt, v.type());
    ASSERT_STREQ("oskar::DoubleRangeExt", v.typeName());

    bool ok = false;
    ASSERT_DOUBLE_EQ(0.0, v.toDouble(&ok));
    ASSERT_TRUE(ok);

    v.toDoubleRangeExt().fromString("all");
    ASSERT_DOUBLE_EQ(0.0-DBL_MIN, v.toDouble(&ok));
    ASSERT_TRUE(ok);
}

TEST(Variant, IntList)
{
    using oskar::IntList;
    IntList l;
    l << 1 << 2;
    Variant v = l;
    ASSERT_EQ(Variant::IntList, v.type());
    ASSERT_STREQ("oskar::IntList", v.typeName());
    ASSERT_EQ((size_t)2, v.toIntList().size());

    ASSERT_EQ(2, v.toIntList().at(1));
    v.toIntList().set(1, 3);
    ASSERT_EQ(3, v.toIntList().at(1));

    v.toIntList().clear();
    ASSERT_EQ((size_t)0, v.toIntList().size());
    bool ok = false;
    v.toIntList().fromString("5,6,7,8,9", ',', &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ((size_t)5, v.toIntList().size());
    ASSERT_EQ(7, v.toIntList().at(2));
    v.toIntList().set(1, -1);
    ASSERT_STREQ("5,-1,7,8,9", v.toString().c_str());
}

TEST(Variant, IntList_extended)
{

    // TODO only allow a certain set of extended values?
    // TODO better constructors

    using oskar::IntList;
    IntList l;
    l.addAllowedString("hello");

    ASSERT_FALSE(l.isValid());
    l << 1 << 2;
    ASSERT_EQ((size_t)2, l.size());
    ASSERT_TRUE(l.isList());
    Variant v = l;
    ASSERT_EQ(Variant::IntList, v.type());
    ASSERT_STREQ("oskar::IntList", v.typeName());
    ASSERT_EQ((size_t)2, v.toIntList().size());

    ASSERT_EQ(2, v.toIntList().at(1));

    v.toIntList() << 5;
    ASSERT_EQ((size_t)3, v.toIntList().size());
    v.toIntList().set(0, -2);
    ASSERT_STREQ("-2,2,5", v.toIntList().toString().c_str());

    bool ok = true;
    v.toIntList().fromString("hello", &ok);
    ASSERT_TRUE(ok);
    ASSERT_FALSE(l.isText());
    ASSERT_STREQ("hello", v.toIntList().toString().c_str());
    ASSERT_EQ((size_t)1, v.toIntList().size());

    v.toIntList().clear();
    ASSERT_EQ((size_t)0, v.toIntList().size());

    v.toIntList().fromString("7,8,9,10,11,-5", ',', &ok);
    ASSERT_TRUE(ok);
    ASSERT_EQ((size_t)6, v.toIntList().size());
    ASSERT_EQ(7, v.toIntList()[0]);
    ASSERT_EQ(8, v.toIntList()[1]);
    ASSERT_EQ(9, v.toIntList()[2]);
    ASSERT_EQ(10, v.toIntList()[3]);
    ASSERT_EQ(11, v.toIntList()[4]);
    ASSERT_EQ(-5, v.toIntList()[5]);

    v.toIntList().fromString("hello", ',', &ok);
    ASSERT_EQ((size_t)1, v.toIntList().size());
    ASSERT_STREQ("hello", v.toIntList().toString().c_str());

    // TODO other accessors
}

TEST(Variant, DoubleList)
{
    using oskar::DoubleList;
    {
        DoubleList d;
        d << 1.1 << 2.2 << 3.3;
        Variant v = d;
        ASSERT_EQ(Variant::DoubleList, v.type());
        ASSERT_STREQ("oskar::DoubleList", v.typeName());
        ASSERT_EQ((size_t)3, v.toDoubleList().size());
        ASSERT_DOUBLE_EQ(1.1, v.toDoubleList()[0]);
        ASSERT_DOUBLE_EQ(2.2, v.toDoubleList()[1]);
        ASSERT_DOUBLE_EQ(3.3, v.toDoubleList()[2]);
        ASSERT_STREQ("1.1,2.2,3.3", v.toString().c_str());
    }

    {
        //Variant v(Variant::DoubleList);
        Variant v(DoubleList(""));
        ASSERT_FALSE(v.isSet()); // This is not set as its empty!
        ASSERT_STREQ("", v.toString().c_str());
        ASSERT_EQ((size_t)0, v.toDoubleList().size());
    }

    {
        Variant v(Variant::DoubleList);
        //ASSERT_FALSE(v.isValid());
        //ASSERT_STREQ("", v.toStr());
        //ASSERT_EQ((size_t)0, v.toDoubleList().size());
    }

    // TODO other accessors
}

TEST(Variant, StringList)
{
    using oskar::StringList;
    StringList l;
    l << "hello" << "again" << "test";
    Variant v = l;
    ASSERT_EQ(Variant::StringList, v.type());
    ASSERT_STREQ("oskar::StringList", v.typeName());
    ASSERT_EQ((size_t)3, v.toStringList().size());
    ASSERT_STREQ("again", v.toStringList()[1].c_str());
    v.toStringList().clear();
    ASSERT_EQ((size_t)0, v.toStringList().size());
}

TEST(Variant, std_string)
{
    // Holding a std::string value
    string stringValue = "5.123";
    Variant v(stringValue);
    ASSERT_EQ(Variant::String, v.type());

    ASSERT_STREQ(stringValue.c_str(), v.toString().c_str());

    // Conversion functions.
    bool ok = true;
    // The string represents a double and therefore this is not currently
    // allowed to cast to int, resulting in a return value of 0 and
    // conversion status of false.
    ASSERT_EQ(0, v.toInt(&ok));
    ASSERT_FALSE(ok);
    ASSERT_STREQ(stringValue.c_str(), v.toString().c_str());
    ASSERT_DOUBLE_EQ(5.123, v.toDouble(&ok));
    ASSERT_TRUE(ok);
    // TODO others?

    // Assignment operator
    Variant other = v;
    ASSERT_EQ(other.type(), v.type());
    ASSERT_EQ(other.toString(), v.toString());

    // Copy constructor
    Variant copy(v);
    ASSERT_EQ(copy.type(), v.type());
    ASSERT_EQ(copy.toString(), v.toString());
}

TEST(Variant, OptionList)
{
    using oskar::OptionList;
    OptionList l("one,two,three");
    Variant v = l;
    ASSERT_EQ(Variant::OptionList, v.type());
    ASSERT_STREQ("oskar::OptionList", v.typeName());
    ASSERT_EQ(3, v.toOptionList().num_options());
    ASSERT_EQ("two", v.toOptionList().option(1));
    bool ok = false;
    v.toOptionList().fromString("three", &ok);
    ASSERT_EQ(2, v.toOptionList().valueIndex());
    ASSERT_TRUE(ok);
    ASSERT_STREQ("three", v.toOptionList().toString().c_str());
    ASSERT_STREQ("three", v.toString().c_str());
    v.fromString("one");
    ASSERT_STREQ("one", v.toOptionList().toString().c_str());
}
