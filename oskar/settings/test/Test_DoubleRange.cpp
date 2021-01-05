/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <climits>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, DoubleRange)
{
    DoubleRange r;
    {
        ASSERT_TRUE(r.init("MIN,MAX"));
        ASSERT_TRUE(r.set_default("0.999999999999"));
        ASSERT_TRUE(r.is_default());
        ASSERT_TRUE(r.set_value("2.3"));
        // FIXME(BM) better double to string conversion.
        EXPECT_STREQ("2.3", r.get_value());
        ASSERT_DOUBLE_EQ(2.3, r.value());
        ASSERT_DOUBLE_EQ(0.999999999999, r.default_value());
        ASSERT_DOUBLE_EQ(-DBL_MAX, r.min());
        ASSERT_DOUBLE_EQ(DBL_MAX, r.max());
    }
    {
        ASSERT_TRUE(r.init("5.4, 10.0"));
        ASSERT_DOUBLE_EQ(5.4, r.min());
        ASSERT_DOUBLE_EQ(10.0, r.max());
        ASSERT_FALSE(r.set_value("3.5"));
        ASSERT_DOUBLE_EQ(5.4, r.value());
        ASSERT_FALSE(r.set_value("311.5"));
        ASSERT_DOUBLE_EQ(10.0, r.value());
        ASSERT_TRUE(r.set_value("7.5"));
        ASSERT_DOUBLE_EQ(7.5, r.value());
        ASSERT_FALSE(r.is_default());
    }

    // Comparison.
    {
        DoubleRange r1, r2;
        ASSERT_TRUE(r1.init("MIN,MAX"));
        ASSERT_TRUE(r2.init("MIN,MAX"));
        ASSERT_TRUE(r1.set_value("2.3"));
        ASSERT_TRUE(r2.set_value("4.5"));
        ASSERT_FALSE(r1 == r2);
        ASSERT_TRUE(r2 > r1);
    }
}

