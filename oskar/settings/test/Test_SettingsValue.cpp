/*
 * Copyright (c) 2015-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "settings/oskar_SettingsValue.h"
#include "settings/oskar_settings_types.h"


TEST(SettingsValue, type_id)
{
    typedef oskar::SettingsValue SV;
    ASSERT_EQ(SV::BOOL, SV::type_id("Bool"));
    ASSERT_EQ(SV::DATE_TIME, SV::type_id("DateTime"));
    ASSERT_EQ(SV::DOUBLE, SV::type_id("Double"));
    ASSERT_EQ(SV::DOUBLE_LIST, SV::type_id("DoubleList"));
    ASSERT_EQ(SV::DOUBLE_RANGE, SV::type_id("DoubleRange"));
    ASSERT_EQ(SV::DOUBLE_RANGE_EXT, SV::type_id("DoubleRangeExt"));
    ASSERT_EQ(SV::INPUT_DIRECTORY, SV::type_id("InputDirectory"));
    ASSERT_EQ(SV::INPUT_FILE, SV::type_id("InputFile"));
    ASSERT_EQ(SV::INPUT_FILE_LIST, SV::type_id("InputFileList"));
    ASSERT_EQ(SV::INT, SV::type_id("Int"));
    ASSERT_EQ(SV::INT_LIST, SV::type_id("IntList"));
    ASSERT_EQ(SV::INT_LIST_EXT, SV::type_id("IntListExt"));
    ASSERT_EQ(SV::INT_POSITIVE, SV::type_id("IntPositive"));
    ASSERT_EQ(SV::INT_RANGE, SV::type_id("IntRange"));
    ASSERT_EQ(SV::INT_RANGE_EXT, SV::type_id("IntRangeExt"));
    ASSERT_EQ(SV::OPTION_LIST, SV::type_id("OptionList"));
    ASSERT_EQ(SV::OUTPUT_FILE, SV::type_id("OutputFile"));
    ASSERT_EQ(SV::RANDOM_SEED, SV::type_id("RandomSeed"));
    ASSERT_EQ(SV::STRING, SV::type_id("String"));
    ASSERT_EQ(SV::STRING_LIST, SV::type_id("StringList"));
    ASSERT_EQ(SV::TIME, SV::type_id("Time"));
    ASSERT_EQ(SV::UNSIGNED_DOUBLE, SV::type_id("UnsignedDouble"));
    ASSERT_EQ(SV::UNSIGNED_INT, SV::type_id("UnsignedInt"));
}


TEST(SettingsValue, type_name)
{
    typedef oskar::SettingsValue SV;
    ASSERT_STREQ("Bool", SV::type_name(SV::BOOL));
    ASSERT_STREQ("DateTime", SV::type_name(SV::DATE_TIME));
    ASSERT_STREQ("Double", SV::type_name(SV::DOUBLE));
    ASSERT_STREQ("DoubleList", SV::type_name(SV::DOUBLE_LIST));
    ASSERT_STREQ("DoubleRange", SV::type_name(SV::DOUBLE_RANGE));
    ASSERT_STREQ("DoubleRangeExt", SV::type_name(SV::DOUBLE_RANGE_EXT));
    ASSERT_STREQ("InputDirectory", SV::type_name(SV::INPUT_DIRECTORY));
    ASSERT_STREQ("InputFile", SV::type_name(SV::INPUT_FILE));
    ASSERT_STREQ("InputFileList", SV::type_name(SV::INPUT_FILE_LIST));
    ASSERT_STREQ("Int", SV::type_name(SV::INT));
    ASSERT_STREQ("IntList", SV::type_name(SV::INT_LIST));
    ASSERT_STREQ("IntListExt", SV::type_name(SV::INT_LIST_EXT));
    ASSERT_STREQ("IntPositive", SV::type_name(SV::INT_POSITIVE));
    ASSERT_STREQ("IntRange", SV::type_name(SV::INT_RANGE));
    ASSERT_STREQ("IntRangeExt", SV::type_name(SV::INT_RANGE_EXT));
    ASSERT_STREQ("OptionList", SV::type_name(SV::OPTION_LIST));
    ASSERT_STREQ("OutputFile", SV::type_name(SV::OUTPUT_FILE));
    ASSERT_STREQ("RandomSeed", SV::type_name(SV::RANDOM_SEED));
    ASSERT_STREQ("String", SV::type_name(SV::STRING));
    ASSERT_STREQ("StringList", SV::type_name(SV::STRING_LIST));
    ASSERT_STREQ("Time", SV::type_name(SV::TIME));
    ASSERT_STREQ("UnsignedDouble", SV::type_name(SV::UNSIGNED_DOUBLE));
    ASSERT_STREQ("UnsignedInt", SV::type_name(SV::UNSIGNED_INT));
}


TEST(SettingsValue, Bool)
{
    {
        oskar::SettingsValue val;
        oskar::Bool b;
        ASSERT_TRUE(b.set_default("false"));
        ASSERT_TRUE(b.set_value("true"));
    }
    {
        oskar::SettingsValue val;
        ASSERT_EQ(oskar::SettingsValue::UNDEF, val.type());
        ASSERT_STREQ("Undef", val.type_name());

        ASSERT_TRUE(val.init("Bool", ""));
        ASSERT_EQ(oskar::SettingsValue::BOOL, val.type());
        ASSERT_TRUE(val.is_default());
        ASSERT_TRUE(val.set_default("false"));

        ASSERT_TRUE(val.set_value("true"));
        ASSERT_FALSE(val.is_default());
        ASSERT_STREQ("true", val.get_value());

        ASSERT_TRUE(val.set_value("false"));
        ASSERT_TRUE(val.is_default());
        ASSERT_STREQ("false", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("Bool", ""));
        ASSERT_TRUE(val1.set_value("false"));
        ASSERT_TRUE(val2.init("Bool", ""));
        ASSERT_TRUE(val2.set_value("true"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, DateTime)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("DateTime", ""));
        ASSERT_EQ(oskar::SettingsValue::DATE_TIME, val.type());
        ASSERT_TRUE(val.set_value("1985-5-23T5:6:12.12345"));
        ASSERT_EQ(1985, val.get<oskar::DateTime>().value().year);
        ASSERT_EQ(
                oskar::DateTime::ISO, val.get<oskar::DateTime>().value().style
        );
        ASSERT_STREQ("", val.get_default());
        ASSERT_STREQ("1985-05-23T05:06:12.12345", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("DateTime", ""));
        ASSERT_TRUE(val1.set_value("2000-01-01T12:00:00"));
        ASSERT_TRUE(val2.init("DateTime", ""));
        ASSERT_TRUE(val2.set_value("2000-01-01T12:00:01"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, Double)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("Double", ""));
        ASSERT_EQ(oskar::SettingsValue::DOUBLE, val.type());
        ASSERT_STREQ("Double", val.type_name());
        ASSERT_TRUE(val.set_default("2.0"));
        ASSERT_TRUE(val.is_default());
        ASSERT_TRUE(val.set_value("0.0"));
        ASSERT_FALSE(val.is_default());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("Double", ""));
        ASSERT_TRUE(val1.set_value("2.1"));
        ASSERT_TRUE(val2.init("Double", ""));
        ASSERT_TRUE(val2.set_value("2.2"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, DoubleList)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("DoubleList", ""));
        ASSERT_EQ(oskar::SettingsValue::DOUBLE_LIST, val.type());
        ASSERT_STREQ("DoubleList", val.type_name());
        ASSERT_TRUE(val.set_value("0.2,   0.4,   0.6"));
        ASSERT_STREQ("0.2,0.4,0.6", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("DoubleList", ""));
        ASSERT_TRUE(val1.set_value("2.2,2.4"));
        ASSERT_TRUE(val2.init("DoubleList", ""));
        ASSERT_TRUE(val2.set_value("3.2,3.4"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, DoubleRange)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("DoubleRange", "-2.0,3.14"));
        ASSERT_EQ(oskar::SettingsValue::DOUBLE_RANGE, val.type());
        ASSERT_STREQ("DoubleRange", val.type_name());
        ASSERT_TRUE(val.set_value("-1.1"));
        ASSERT_STREQ("-1.1", val.get_value());
        ASSERT_TRUE(val.set_value("2.56"));
        ASSERT_STREQ("2.56", val.get_value());
        ASSERT_FALSE(val.set_value("-2.5"));
        ASSERT_STREQ("-2.0", val.get_value());
        ASSERT_FALSE(val.set_value("3.14159"));
        ASSERT_STREQ("3.14", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("DoubleRange", "-2.0,3.14"));
        ASSERT_TRUE(val1.set_value("-1.1"));
        ASSERT_TRUE(val2.init("DoubleRange", "-20.0,6.28"));
        ASSERT_TRUE(val2.set_value("5.12"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, DoubleRangeExt)
{
    {
        oskar::SettingsValue val;
        bool ok = false;
        ASSERT_TRUE(val.init("DoubleRangeExt", "-MAX,MAX,min,max"));
        ASSERT_TRUE(val.set_default("min"));
        ASSERT_TRUE(val.set_value("10.0"));
        ASSERT_DOUBLE_EQ(10.0, val.to_double(ok));
        ASSERT_TRUE(ok);
        ASSERT_TRUE(val.set_value("min"));
        ASSERT_DOUBLE_EQ(-DBL_MAX, val.to_double(ok));
        ASSERT_STREQ("min", val.to_string());
        ASSERT_TRUE(ok);
        ASSERT_TRUE(val.set_value("max"));
        ASSERT_DOUBLE_EQ(DBL_MAX, val.to_double(ok));
        ASSERT_STREQ("max", val.to_string());
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("DoubleRangeExt", "-MAX,MAX,min,max"));
        ASSERT_TRUE(val1.set_value("-1.1"));
        ASSERT_TRUE(val2.init("DoubleRangeExt", "-MAX,MAX,min,max"));
        ASSERT_TRUE(val2.set_value("5.12"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, InputDirectory)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("InputDirectory", ""));
        ASSERT_EQ(oskar::SettingsValue::INPUT_DIRECTORY, val.type());
        ASSERT_STREQ("InputDirectory", val.type_name());
        ASSERT_TRUE(val.set_value("/a/directory/name"));
        ASSERT_STREQ("/a/directory/name", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("InputDirectory", ""));
        ASSERT_TRUE(val1.set_value("/a/directory/name"));
        ASSERT_TRUE(val2.init("InputDirectory", ""));
        ASSERT_TRUE(val2.set_value("/some/other/directory"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, InputFile)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("InputFile", ""));
        ASSERT_EQ(oskar::SettingsValue::INPUT_FILE, val.type());
        ASSERT_STREQ("InputFile", val.type_name());
        ASSERT_TRUE(val.set_value("a_file_name.dat"));
        ASSERT_STREQ("a_file_name.dat", val.get_value());
        bool ok = false;
        ASSERT_DOUBLE_EQ(0., val.to_double(ok));
        ASSERT_FALSE(ok);
        ASSERT_EQ(0, val.to_int(ok));
        ASSERT_FALSE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("InputFile", ""));
        ASSERT_TRUE(val1.set_value("a_file_name.dat"));
        ASSERT_TRUE(val2.init("InputFile", ""));
        ASSERT_TRUE(val2.set_value("some/other/file.dat"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, InputFileList)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("InputFileList", ""));
        ASSERT_EQ(oskar::SettingsValue::INPUT_FILE_LIST, val.type());
        ASSERT_STREQ("InputFileList", val.type_name());
        ASSERT_TRUE(val.set_value("a_file_name.dat, another_file.dat"));
        ASSERT_STREQ("a_file_name.dat,another_file.dat", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("InputFileList", ""));
        ASSERT_TRUE(val1.set_value("a_file_name.dat, another_file.dat"));
        ASSERT_TRUE(val2.init("InputFileList", ""));
        ASSERT_TRUE(val2.set_value("some/file1.dat, some/file2.dat"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, Int)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("Int", ""));
        ASSERT_EQ(oskar::SettingsValue::INT, val.type());
        ASSERT_STREQ("Int", val.type_name());
        ASSERT_TRUE(val.set_value("-1234"));
        ASSERT_STREQ("-1234", val.get_value());
        bool ok = false;
        ASSERT_EQ(-1234, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_EQ(0u, val.to_unsigned(ok));
        ASSERT_FALSE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("Int", ""));
        ASSERT_TRUE(val1.set_value("12345"));
        ASSERT_TRUE(val2.init("Int", ""));
        ASSERT_TRUE(val2.set_value("54321"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, IntList)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("IntList", ""));
        ASSERT_EQ(oskar::SettingsValue::INT_LIST, val.type());
        ASSERT_STREQ("IntList", val.type_name());
        ASSERT_TRUE(val.set_value("-1234, 321"));
        ASSERT_STREQ("-1234,321", val.get_value());
        bool ok = false;
        int size = 0;
        const int* list = val.to_int_list(&size, ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(2, size);
        ASSERT_EQ(-1234, list[0]);
        ASSERT_EQ(321, list[1]);
        ok = false;
        ASSERT_EQ(-1234, val.to_int(ok));
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("IntList", ""));
        ASSERT_TRUE(val1.set_value("12345,32"));
        ASSERT_TRUE(val2.init("IntList", ""));
        ASSERT_TRUE(val2.set_value("54321,64"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, IntListExt)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("IntListExt", "all"));
        ASSERT_EQ(oskar::SettingsValue::INT_LIST_EXT, val.type());
        ASSERT_STREQ("IntListExt", val.type_name());
        ASSERT_TRUE(val.set_value("-1234, 321"));
        ASSERT_STREQ("-1234,321", val.get_value());
        bool ok = false;
        int size = 0;
        const int* list = val.to_int_list(&size, ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(2, size);
        ASSERT_EQ(-1234, list[0]);
        ASSERT_EQ(321, list[1]);
        ASSERT_TRUE(val.set_value("all"));
        ASSERT_STREQ("all", val.get_value());
        ASSERT_FALSE(val.set_value("blah"));
        ASSERT_STREQ("all", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("IntListExt", "all"));
        ASSERT_TRUE(val1.set_value("12345,32"));
        ASSERT_TRUE(val2.init("IntListExt", "all"));
        ASSERT_TRUE(val2.set_value("54321,64"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, IntPositive)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("IntPositive", ""));
        ASSERT_EQ(oskar::SettingsValue::INT_POSITIVE, val.type());
        ASSERT_STREQ("IntPositive", val.type_name());
        ASSERT_TRUE(val.set_value("12"));
        ASSERT_STREQ("12", val.get_value());
        bool ok = false;
        ASSERT_EQ(12, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_FALSE(val.set_value("0"));
        ASSERT_FALSE(val.set_value("-1"));
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("IntPositive", ""));
        ASSERT_TRUE(val1.set_value("12345"));
        ASSERT_TRUE(val2.init("IntPositive", ""));
        ASSERT_TRUE(val2.set_value("54321"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, IntRange)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("IntRange", "-2,3"));
        ASSERT_EQ(oskar::SettingsValue::INT_RANGE, val.type());
        ASSERT_STREQ("IntRange", val.type_name());
        ASSERT_TRUE(val.set_value("-1"));
        ASSERT_STREQ("-1", val.get_value());
        bool ok = false;
        ASSERT_EQ(-1, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_TRUE(val.set_value("2"));
        ASSERT_STREQ("2", val.get_value());
        ASSERT_EQ(2, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_FALSE(val.set_value("-4"));
        ASSERT_STREQ("-2", val.get_value());
        ASSERT_EQ(-2, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_FALSE(val.set_value("5"));
        ASSERT_STREQ("3", val.get_value());
        ASSERT_EQ(3, val.to_int(ok));
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("IntRange", "-2,3"));
        ASSERT_TRUE(val1.set_value("-1"));
        ASSERT_TRUE(val2.init("IntRange", "-20,6"));
        ASSERT_TRUE(val2.set_value("5"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, IntRangeExt)
{
    {
        oskar::SettingsValue val;
        bool ok = false;
        ASSERT_TRUE(val.init("IntRangeExt", "-15,10,min"));
        ASSERT_TRUE(val.set_default("min"));
        ASSERT_TRUE(val.set_value("10"));
        ASSERT_EQ(10, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_TRUE(val.set_value("min"));
        ASSERT_EQ(-15, val.to_int(ok));
        ASSERT_STREQ("min", val.to_string());
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val;
        bool ok = false;
        ASSERT_TRUE(val.init("IntRangeExt", "-15,10,min,max"));
        ASSERT_TRUE(val.set_default("min"));
        ASSERT_TRUE(val.set_value("5"));
        ASSERT_EQ(5, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_TRUE(val.set_value("max"));
        ASSERT_EQ(10, val.to_int(ok));
        ASSERT_STREQ("max", val.to_string());
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("IntRangeExt", "-15,10,min"));
        ASSERT_TRUE(val1.set_value("-1"));
        ASSERT_TRUE(val2.init("IntRangeExt", "-15,10,min"));
        ASSERT_TRUE(val2.set_value("5"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, OptionList)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("OptionList", "alice,bob,charlie"));
        ASSERT_EQ(oskar::SettingsValue::OPTION_LIST, val.type());
        ASSERT_STREQ("OptionList", val.type_name());
        ASSERT_TRUE(val.set_value("alice"));
        ASSERT_STREQ("alice", val.get_value());
        ASSERT_TRUE(val.set_value("charlie"));
        ASSERT_STREQ("charlie", val.get_value());
        ASSERT_FALSE(val.set_value("eve"));
        ASSERT_STREQ("charlie", val.get_value());
        bool ok = false;
        int size = 0;
        ASSERT_FALSE(val.to_int_list(&size, ok));
        ASSERT_FALSE(ok);
        ASSERT_FALSE(val.to_double_list(&size, ok));
        ASSERT_FALSE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("OptionList", "alice,bob,charlie"));
        ASSERT_TRUE(val1.set_value("bob"));
        ASSERT_TRUE(val2.init("OptionList", "eve,frank"));
        ASSERT_TRUE(val2.set_value("eve"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, OutputFile)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("OutputFile", ""));
        ASSERT_EQ(oskar::SettingsValue::OUTPUT_FILE, val.type());
        ASSERT_STREQ("OutputFile", val.type_name());
        ASSERT_TRUE(val.set_value("a_file_name.dat"));
        ASSERT_STREQ("a_file_name.dat", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("OutputFile", ""));
        ASSERT_TRUE(val1.set_value("a_file_name.dat"));
        ASSERT_TRUE(val2.init("OutputFile", ""));
        ASSERT_TRUE(val2.set_value("some/other/file.dat"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, RandomSeed)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("RandomSeed", ""));
        ASSERT_EQ(oskar::SettingsValue::RANDOM_SEED, val.type());
        ASSERT_STREQ("RandomSeed", val.type_name());
        ASSERT_TRUE(val.set_value("12"));
        ASSERT_STREQ("12", val.get_value());
        bool ok = false;
        ASSERT_EQ(12, val.to_int(ok));
        ASSERT_TRUE(ok);
        ASSERT_FALSE(val.set_value("0"));
        ASSERT_FALSE(val.set_value("-1"));
        int size = 0;
        ASSERT_FALSE(val.to_string_list(&size, ok));
        ASSERT_FALSE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("RandomSeed", ""));
        ASSERT_TRUE(val1.set_value("12345"));
        ASSERT_TRUE(val2.init("RandomSeed", ""));
        ASSERT_TRUE(val2.set_value("54321"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, String)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("String", ""));
        ASSERT_EQ(oskar::SettingsValue::STRING, val.type());
        ASSERT_STREQ("String", val.type_name());
        ASSERT_TRUE(val.set_value("the quick brown fox"));
        ASSERT_STREQ("the quick brown fox", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("String", ""));
        ASSERT_TRUE(val1.set_value("the quick brown fox"));
        ASSERT_TRUE(val2.init("String", ""));
        ASSERT_TRUE(val2.set_value("x-ray"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, StringList)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("StringList", ""));
        ASSERT_EQ(oskar::SettingsValue::STRING_LIST, val.type());
        ASSERT_STREQ("StringList", val.type_name());
        ASSERT_TRUE(val.set_value("the quick brown fox, jumps over"));
        ASSERT_STREQ("the quick brown fox,jumps over", val.get_value());
        bool ok = false;
        int size = 0;
        const char* const* list = val.to_string_list(&size, ok);
        ASSERT_TRUE(ok);
        ASSERT_EQ(2, size);
        ASSERT_STREQ("the quick brown fox", list[0]);
        ASSERT_STREQ("jumps over", list[1]);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("StringList", ""));
        ASSERT_TRUE(val1.set_value("the quick brown fox, jumps over"));
        ASSERT_TRUE(val2.init("StringList", ""));
        ASSERT_TRUE(val2.set_value("x-ray,yankee"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, Time)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("Time", ""));
        ASSERT_EQ(oskar::SettingsValue::TIME, val.type());
        ASSERT_TRUE(val.set_value("05:6:12.12345"));
        ASSERT_EQ(5, val.get<oskar::Time>().value().hours);
        ASSERT_EQ(6, val.get<oskar::Time>().value().minutes);
        ASSERT_STREQ("", val.get_default());
        ASSERT_STREQ("05:06:12.12345", val.get_value());
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("Time", ""));
        ASSERT_TRUE(val1.set_value("12:00:00"));
        ASSERT_TRUE(val2.init("Time", ""));
        ASSERT_TRUE(val2.set_value("12:00:01"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, UnsignedDouble)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("UnsignedDouble", ""));
        ASSERT_EQ(oskar::SettingsValue::UNSIGNED_DOUBLE, val.type());
        ASSERT_STREQ("UnsignedDouble", val.type_name());
        ASSERT_TRUE(val.set_value("1234.56"));
        ASSERT_STREQ("1234.56", val.get_value());
        bool ok = false;
        ASSERT_DOUBLE_EQ(1234.56, val.to_double(ok));
        ASSERT_TRUE(ok);
        ASSERT_FALSE(val.set_value("-1234.56"));
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("UnsignedDouble", ""));
        ASSERT_TRUE(val1.set_value("12345.67"));
        ASSERT_TRUE(val2.init("UnsignedDouble", ""));
        ASSERT_TRUE(val2.set_value("76543.21"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}


TEST(SettingsValue, UnsignedInt)
{
    {
        oskar::SettingsValue val;
        ASSERT_TRUE(val.init("UnsignedInt", ""));
        ASSERT_EQ(oskar::SettingsValue::UNSIGNED_INT, val.type());
        ASSERT_STREQ("UnsignedInt", val.type_name());
        ASSERT_TRUE(val.set_value("1234"));
        ASSERT_STREQ("1234", val.get_value());
        bool ok = false;
        ASSERT_EQ(1234u, val.to_unsigned(ok));
        ASSERT_TRUE(ok);
    }
    {
        oskar::SettingsValue val1, val2;
        ASSERT_TRUE(val1.init("UnsignedInt", ""));
        ASSERT_TRUE(val1.set_value("12345"));
        ASSERT_TRUE(val2.init("UnsignedInt", ""));
        ASSERT_TRUE(val2.set_value("54321"));
        ASSERT_TRUE(val1 != val2);
        ASSERT_FALSE(val1 == val2);
        ASSERT_TRUE(val1 < val2);
        ASSERT_TRUE(val1 <= val2);
        ASSERT_FALSE(val1 > val2);
        ASSERT_FALSE(val1 >= val2);
    }
}
