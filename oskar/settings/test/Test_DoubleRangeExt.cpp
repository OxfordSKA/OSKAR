/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;
using namespace std;

TEST(settings_types, DoubleRangeExt)
{
    DoubleRangeExt r;
    {
        // This should fail as there is no extended string
        ASSERT_FALSE(r.init("2.0,5.0"));
        ASSERT_STREQ("0.0", r.get_default());
    }
    {
        ASSERT_TRUE(r.init("2.0,5.0,min"));
        ASSERT_STREQ("0.0", r.get_default());
        ASSERT_TRUE(r.set_default("3.2"));
        ASSERT_TRUE(r.is_default());
        // FIXME(BM) double to string printing...
        EXPECT_STREQ("3.2", r.get_default());
        EXPECT_STREQ("3.2", r.get_value());
        ASSERT_DOUBLE_EQ(3.2, r.value());
        ASSERT_TRUE(r.set_value("5.1"));
        ASSERT_STREQ("5.0", r.get_value());
        ASSERT_TRUE(r.set_value("1.1"));
        ASSERT_STREQ("min", r.get_value());
        ASSERT_TRUE(r.set_value("2.1234567891"));
        ASSERT_STREQ("2.1234567891", r.get_value());
        ASSERT_DOUBLE_EQ(2.0, r.min());
        ASSERT_DOUBLE_EQ(5.0, r.max());
        ASSERT_STREQ("min", r.ext_min());
        ASSERT_STREQ("", r.ext_max());
    }
    {
        ASSERT_TRUE(r.init("2.0,5.0,min, max"));
        ASSERT_STREQ("0.0", r.get_value());
        ASSERT_STREQ("0.0", r.get_default());
        ASSERT_TRUE(r.set_value("3.0"));
        ASSERT_STREQ("3.0", r.get_value());
        ASSERT_TRUE(r.set_value("3.53"));
        EXPECT_STREQ("3.53", r.get_value());
        ASSERT_DOUBLE_EQ(3.53, r.value());
        ASSERT_TRUE(r.set_value("5.1"));
        ASSERT_STREQ("max", r.get_value());
        ASSERT_TRUE(r.set_value("1.1"));
        ASSERT_STREQ("min", r.get_value());
        ASSERT_TRUE(r.set_value("2.1234567891"));
        ASSERT_STREQ("2.1234567891", r.get_value());
    }
    {
        ASSERT_TRUE(r.init("MIN,MAX,MAX"));
    }
    {
        ASSERT_TRUE(r.init("-MAX,MAX,min,max"));
        ASSERT_TRUE(r.set_default("min"));
        ASSERT_STREQ("min", r.get_value());
    }
    {
        ASSERT_TRUE(r.init("-MAX,MAX,min,max"));
        ASSERT_TRUE(r.set_default("max"));
        ASSERT_STREQ("max", r.get_value());
        r.set_value("10.0");
        ASSERT_STREQ("10.0", r.get_value());
        ASSERT_STREQ("min", r.ext_min());
        ASSERT_STREQ("max", r.ext_max());
    }

    // Comparison.
    {
        DoubleRangeExt r1, r2;
        ASSERT_TRUE(r1.init("2.0,5.0,min"));
        ASSERT_TRUE(r2.init("2.0,5.0,min"));
        ASSERT_TRUE(r1.set_value("2.345"));
        ASSERT_TRUE(r2.set_value("3.456"));
        ASSERT_FALSE(r1 == r2);
        ASSERT_TRUE(r2 > r1);
    }
}

