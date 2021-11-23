/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include <iostream>
#include <climits>
#include <limits>
#include "settings/oskar_settings_utility_string.h"
#include <cmath>

using namespace std;

TEST(oskar_settings_utility, string_trim)
{
    string s;
    s = "   hello there\t  ";
    ASSERT_STREQ("hello there", oskar_settings_utility_string_trim(s).c_str());
}


TEST(oskar_settings_utility, string_get_type_params)
{
    string s;
    vector<string> p;
    {
        s = "2,10";
        p = oskar_settings_utility_string_get_type_params(s);
        ASSERT_EQ(2u, p.size());
        ASSERT_STREQ("2", p[0].c_str());
        ASSERT_STREQ("10", p[1].c_str());
    }
    {
        s = "2,\"10,20\",3";
        p = oskar_settings_utility_string_get_type_params(s);
        ASSERT_EQ(3u, p.size());
        ASSERT_STREQ("2",     p[0].c_str());
        ASSERT_STREQ("10,20", p[1].c_str());
        ASSERT_STREQ("3",     p[2].c_str());
    }
    {
        s = "\n2,\"10,20\",\n3\n";
        p = oskar_settings_utility_string_get_type_params(s);
        ASSERT_EQ(3u, p.size());
        ASSERT_STREQ("2",     p[0].c_str());
        ASSERT_STREQ("10,20", p[1].c_str());
        ASSERT_STREQ("3",     p[2].c_str());
    }
}


TEST(oskar_settings_utility, string_to_int)
{
    string s;
    bool ok = false;
    {
        s = "123456789";
        ASSERT_EQ(123456789, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "";
        ASSERT_EQ(0, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_FALSE(ok);
    }
    {
        s = "-123456789";
        ASSERT_EQ(-123456789, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "2147483647";
        ASSERT_EQ(INT_MAX, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "-2147483647";
        ASSERT_EQ(-INT_MAX, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "   1  ";
        ASSERT_EQ(1, oskar_settings_utility_string_to_int(s, &ok));
        // Note this fails due to checking for trailing characters after the int
        ASSERT_FALSE(ok);
    }
    {
        s = "hello";
        ASSERT_EQ(0, oskar_settings_utility_string_to_int(s, &ok));
        ASSERT_FALSE(ok);
    }
}

TEST(oskar_settings_utility, int_to_string)
{
    int i = 0;
    {
        i = 23;
        ASSERT_STREQ("23", oskar_settings_utility_int_to_string(i).c_str());
    }
    {
        i = -123456790;
        ASSERT_STREQ("-123456790", oskar_settings_utility_int_to_string(i).c_str());
    }
}


TEST(oskar_settings_utility, string_to_upper)
{
    std::string s = "hello there";
    ASSERT_STREQ("HELLO THERE", oskar_settings_utility_string_to_upper(s).c_str());
}

TEST(oskar_settings_utility, string_starts_with)
{
    std::string s1 = "hello there";
    std::string s2;
    {
        s2 = "hello";
        ASSERT_TRUE(oskar_settings_utility_string_starts_with(s1, s2, true));
    }
    {
        s2 = "foo";
        ASSERT_FALSE(oskar_settings_utility_string_starts_with(s1, s2, true));
    }
    {
        s2 = "H";
        ASSERT_FALSE(oskar_settings_utility_string_starts_with(s1, s2, true));
    }
    {
        s2 = "He";
        ASSERT_TRUE(oskar_settings_utility_string_starts_with(s1, s2, false));
    }
    {
        s1 = "TIME";
        s2 = "T";
        ASSERT_TRUE(oskar_settings_utility_string_starts_with(s1, s2, false));
    }
    {
        s1 = "TIME";
        s2 = "t";
        ASSERT_TRUE(oskar_settings_utility_string_starts_with(s1, s2, false));
    }
}

static int get_precision(double value) {
    int n = 17;
    if (value != 0.0 && value > 1.0) {
        n -= (floor(log10(value)) + 1);
    }
    return n;
}

TEST(oskar_settings_utility, double_to_string_2)
{
    {
        double value = 100.0000012341111;
        cout << endl;
        cout << "Input: " << setprecision(17) << value << " " << get_precision(value) << endl;
        double d = value;
        int n = get_precision(value);
        cout << " f: " << oskar_settings_utility_double_to_string_2(d, 'f', n) << endl;
        cout << " e: " << oskar_settings_utility_double_to_string_2(d, 'e', n) << endl;
        cout << " g: " << oskar_settings_utility_double_to_string_2(d, 'g', n) << endl;
    }
    {
        double value = 0.000001234567891011;
        cout << endl;
        cout << "Input: " << setprecision(17) << value << " " << get_precision(value) << endl;
        double d = value;
        int n = get_precision(value);
        cout << " f: " << oskar_settings_utility_double_to_string_2(d, 'f', n) << endl;
        cout << " e: " << oskar_settings_utility_double_to_string_2(d, 'e', n) << endl;
        cout << " g: " << oskar_settings_utility_double_to_string_2(d, 'g', n) << endl;
    }
    {
        double value = 1.1234587891011100e8;
        cout << endl;
        cout << "Input: " << setprecision(17) << value << " " << get_precision(value) << endl;
        double d = value;
        int n = get_precision(value);
        cout << " f: " << oskar_settings_utility_double_to_string_2(d, 'f', n) << endl;
        n = 16;
        cout << " e: " << oskar_settings_utility_double_to_string_2(d, 'e', n) << endl;
        n = 17;
        cout << " g: " << oskar_settings_utility_double_to_string_2(d, 'g', n) << endl;
    }
    {
        double value = 1.234e-6;
        cout << endl;
        cout << "Input: " << setprecision(17) << value << " " << get_precision(value) << endl;
        double d = value;
        int n = get_precision(value);
        cout << " f: " << oskar_settings_utility_double_to_string_2(d, 'f', n) << endl;
        n = 16;
        cout << " e: " << oskar_settings_utility_double_to_string_2(d, 'e', n) << endl;
        n = 17;
        cout << " g: " << oskar_settings_utility_double_to_string_2(d, 'g', n) << endl;
    }
    {
        double value = 1.234e-5;
        cout << endl;
        cout << "Input: " << setprecision(17) << value << " " << get_precision(value) << endl;
        double d = value;
        cout << " f: " << oskar_settings_utility_double_to_string_2(d, 'f') << endl;
        cout << " e: " << oskar_settings_utility_double_to_string_2(d, 'e') << endl;
        cout << " g: " << oskar_settings_utility_double_to_string_2(d, 'g') << endl;
    }

}

TEST(oskar_settings_utility, string_to_double)
{
    string s;
    bool ok = false;
    {
        s = "123456789.0";
        ASSERT_DOUBLE_EQ(123456789.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "";
        ASSERT_DOUBLE_EQ(0.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }
    {
        s = "-123456789";
        ASSERT_DOUBLE_EQ(-123456789.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "1.79769313486231570815e+308";
        ASSERT_DOUBLE_EQ(DBL_MAX, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "1.79769313486231570815e+309";
        ASSERT_DOUBLE_EQ(std::numeric_limits<double>::infinity(),
                oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }

    {
        s = "-1.79769313486231570815e+308";
        ASSERT_DOUBLE_EQ(-DBL_MAX, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_TRUE(ok);
    }
    {
        s = "-1.79769313486231570815e+309";
        ASSERT_EQ(-std::numeric_limits<double>::infinity(),
                oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }
    {
        s = "   1  ";
        ASSERT_DOUBLE_EQ(1.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }
    {
        s = "hello";
        ASSERT_DOUBLE_EQ(0.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }

    {
        s = "5.0foo";
        ASSERT_DOUBLE_EQ(5.0, oskar_settings_utility_string_to_double(s, &ok));
        ASSERT_FALSE(ok);
    }
}
