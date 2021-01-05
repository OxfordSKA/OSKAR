/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, String)
{
    String s;
    ASSERT_TRUE(s.set_default("hello"));
    ASSERT_STREQ("hello", s.get_default());
    ASSERT_STREQ("hello", s.get_value());
    ASSERT_TRUE(s.is_default());
    ASSERT_TRUE(s.set_value("there"));
    ASSERT_FALSE(s.is_default());
    ASSERT_STREQ("there", s.get_value());

    // Comparison.
    {
        String s1, s2;
        ASSERT_TRUE(s1.set_value("hello"));
        ASSERT_TRUE(s2.set_value("hello"));
        ASSERT_TRUE(s1 == s2);
        ASSERT_FALSE(s2 > s1);
    }
}
