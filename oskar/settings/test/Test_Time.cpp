/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, Time)
{
    Time t;
    EXPECT_TRUE(t.init(""));
    EXPECT_STREQ("", t.get_value());

    EXPECT_TRUE(t.set_value("0.1"));
    EXPECT_STREQ("0.1", t.get_value());
    EXPECT_DOUBLE_EQ(0.1, t.to_seconds());

    EXPECT_TRUE(t.set_default("1:2:3.45678"));
    EXPECT_EQ(1, t.value().hours);
    EXPECT_EQ(2, t.value().minutes);
    EXPECT_DOUBLE_EQ(3.45678, t.value().seconds);
    EXPECT_STREQ("01:02:03.45678", t.get_default());
    EXPECT_TRUE(t.is_default());

    EXPECT_TRUE(t.set_value("01:02:23.45678"));
    EXPECT_FALSE(t.is_default());
    EXPECT_EQ(1, t.value().hours);
    EXPECT_EQ(2, t.value().minutes);
    EXPECT_DOUBLE_EQ(23.45678, t.value().seconds);
    EXPECT_STREQ("01:02:23.45678", t.get_value());

    EXPECT_TRUE(t.set_value("123.4567891011"));
    EXPECT_FALSE(t.is_default());
    EXPECT_EQ(0, t.value().hours);
    EXPECT_EQ(2, t.value().minutes);
    EXPECT_NEAR(3.4567891011, t.value().seconds, 1e-8);
    EXPECT_DOUBLE_EQ(123.4567891011, t.to_seconds());
    EXPECT_STREQ("123.4567891011", t.get_value());

    // EXPECT_STREQ("00:02:03.4567891011", t.get_value());

    EXPECT_TRUE(t.set_value("1.234567891011121e+04"));
    EXPECT_EQ(3, t.value().hours);
    EXPECT_EQ(25, t.value().minutes);
    EXPECT_NEAR(45.67891011121, t.value().seconds, 1e-8);
    EXPECT_DOUBLE_EQ(1.234567891011121e+04, t.to_seconds());
    EXPECT_STREQ("12345.67891011121", t.get_value());
    //EXPECT_STREQ("03:25:45.67891011121", t.get_value());

    EXPECT_TRUE(t.set_value("23:02:23.45678"));

    EXPECT_TRUE(t.set_value("23:59:59.9"));
    EXPECT_EQ(23, t.value().hours);
    EXPECT_EQ(59, t.value().minutes);
    EXPECT_DOUBLE_EQ(59.9, t.value().seconds);

    EXPECT_TRUE(t.set_value("24:00:59.9"));
    EXPECT_TRUE(t.set_value("48:00:59.9"));
    EXPECT_TRUE(t.set_value("100:00:59.9"));
    EXPECT_EQ(100, t.value().hours);
    EXPECT_EQ(0, t.value().minutes);
    EXPECT_DOUBLE_EQ(59.9, t.value().seconds);
    EXPECT_FALSE(t.set_value("23:60:59.9"));
    EXPECT_FALSE(t.set_value("23:59:60.0"));

    // Comparison.
    {
        Time t1, t2;
        EXPECT_TRUE(t1.set_value("01:02:03.45678"));
        EXPECT_TRUE(t2.set_value("01:02:03.45678"));
        EXPECT_TRUE(t1 == t2);
        EXPECT_FALSE(t2 > t1);
        EXPECT_TRUE(t2.set_value("02:02:03.45678"));
        EXPECT_FALSE(t1 == t2);
        EXPECT_TRUE(t2 > t1);
    }
}
