/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, IntRange)
{
    IntRange r;
    ASSERT_STREQ("0", r.get_default());
    ASSERT_STREQ("0", r.get_value());
    ASSERT_TRUE(r.is_default());
    ASSERT_TRUE(r.init("2,5"));
    ASSERT_EQ(2, r.min());
    ASSERT_EQ(5, r.max());
    ASSERT_TRUE(r.set_default("2"));
    ASSERT_EQ(2, r.default_value());
    ASSERT_TRUE(r.is_default());
    ASSERT_TRUE(r.set_value("4"));
    ASSERT_EQ(4, r.value());
    ASSERT_STREQ("4", r.get_value());
    ASSERT_FALSE(r.is_default());
    ASSERT_FALSE(r.set_value("6"));
    ASSERT_FALSE(r.set_value("2.111"));

    // Comparison.
    {
        IntRange r1, r2;
        ASSERT_TRUE(r1.init("2,5"));
        ASSERT_TRUE(r2.init("2,5"));
        ASSERT_TRUE(r1.set_value("2"));
        ASSERT_TRUE(r2.set_value("3"));
        ASSERT_FALSE(r1 == r2);
        ASSERT_TRUE(r2 > r1);
    }
}
