/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, Double)
{
    Double d;
    ASSERT_TRUE(d.set_default("1.5"));
    ASSERT_STREQ("1.5", d.get_default());
    ASSERT_STREQ("1.5", d.get_value());
    ASSERT_TRUE(d.is_default());
    ASSERT_TRUE(d.set_value("9.123456789e23"));
    ASSERT_FALSE(d.is_default());
    ASSERT_EQ(9.123456789e23, d.value());
    // FIXME(BM) exponential format for string return for very large numbers
    ASSERT_EQ(1.5, d.default_value());

    ASSERT_TRUE(d.set_value("0.1234567e6"));
    ASSERT_DOUBLE_EQ(0.1234567e6, d.value());
    ASSERT_STREQ("1.234567e+05", d.get_value());

    ASSERT_TRUE(d.set_value("-12.12345"));
    ASSERT_DOUBLE_EQ(-12.12345, d.value());

    // Comparison.
    {
        Double d1, d2;
        ASSERT_TRUE(d1.set_value("12.345"));
        ASSERT_TRUE(d2.set_value("34.567"));
        ASSERT_FALSE(d1 == d2);
        ASSERT_TRUE(d2 > d1);
    }
}

