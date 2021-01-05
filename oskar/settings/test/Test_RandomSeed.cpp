/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, RandomSeed)
{
    RandomSeed s;
    ASSERT_STREQ("1", s.get_value());
    ASSERT_TRUE(s.init(""));
    ASSERT_FALSE(s.set_default("0"));
    ASSERT_EQ(1, s.default_value());
    ASSERT_TRUE(s.set_default("time"));
    ASSERT_TRUE(s.is_default());
    ASSERT_STREQ("time", s.get_default());
    ASSERT_TRUE(s.set_value("12345"));
    ASSERT_FALSE(s.is_default());
    ASSERT_STREQ("12345", s.get_value());
    ASSERT_EQ(12345, s.value());
    ASSERT_TRUE(s.set_value("t"));
    ASSERT_STREQ("time", s.get_value());
    ASSERT_TRUE(s.set_value("Time"));
    ASSERT_STREQ("time", s.get_value());

    // Comparison.
    {
        RandomSeed r1, r2;
        ASSERT_TRUE(r1.init(""));
        ASSERT_TRUE(r2.init(""));
        ASSERT_TRUE(r1.set_value("2"));
        ASSERT_TRUE(r2.set_value("3"));
        ASSERT_FALSE(r1 == r2);
        ASSERT_TRUE(r2 > r1);
    }
}
