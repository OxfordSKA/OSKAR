/*
 * Copyright (c) 2014-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"
#include <climits>

using namespace oskar;

TEST(settings_types, IntRangeExt)
{
    IntRangeExt i;
    ASSERT_STREQ("0", i.get_default());
    ASSERT_TRUE(i.init("0,MAX,smin"));
    ASSERT_EQ(0, i.min());
    ASSERT_EQ(INT_MAX, i.max());
    ASSERT_STREQ("smin", i.ext_min());
    ASSERT_TRUE(i.set_default("12"));
    ASSERT_EQ(12, i.value());
    ASSERT_STREQ("12", i.get_default());
    ASSERT_STREQ("12", i.get_value());
    ASSERT_TRUE(i.is_default());
    ASSERT_FALSE(i.set_value("-1"));
    ASSERT_STREQ("12", i.get_value());
    ASSERT_TRUE(i.is_default());
    ASSERT_TRUE(i.set_value("10"));
    ASSERT_STREQ("10", i.get_value());

    // Not allowed as 'smax' would be unable to map
    // to a unique integer value.
    ASSERT_FALSE(i.init("0,MAX,smin,smax"));
    ASSERT_TRUE(i.init("0,100,smin,smax"));
    ASSERT_TRUE(i.set_default("10"));
    ASSERT_STREQ("10", i.get_default());
    ASSERT_FALSE(i.set_value("101")); // fails because 101 is outside the range.
    ASSERT_STREQ("10", i.get_value());
    ASSERT_FALSE(i.set_value("-1")); // fails because -1 is outside the range.
    ASSERT_STREQ("10", i.get_value());
    ASSERT_TRUE(i.set_value("smin"));
    ASSERT_STREQ("smin", i.get_value());
    ASSERT_TRUE(i.set_value("smax"));
    ASSERT_STREQ("smax", i.get_value());

    ASSERT_TRUE(i.init("0,10,smin"));
    ASSERT_FALSE(i.set_value("11"));
    ASSERT_EQ(0, i.value());
    ASSERT_STREQ("smin", i.get_value());

    ASSERT_TRUE(i.init("-1,INT_MAX,smin"));
    ASSERT_TRUE(i.set_default("5"));
    ASSERT_EQ(5, i.value());
    ASSERT_TRUE(i.set_value("smin"));
    ASSERT_STREQ("smin", i.get_value());

    ASSERT_TRUE(i.init("0,MAX,auto"));
    ASSERT_TRUE(i.set_default("auto"));
    ASSERT_TRUE(i.set_value("0"));
    ASSERT_EQ(0, i.value());
    ASSERT_STREQ("auto", i.get_value());
    ASSERT_TRUE(i.is_default());
}
