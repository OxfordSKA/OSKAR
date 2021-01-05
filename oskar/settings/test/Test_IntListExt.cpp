/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, IntListExt)
{
    IntListExt l;
    ASSERT_TRUE(l.init("all"));
    ASSERT_STREQ("all", l.special_string());
    ASSERT_TRUE(l.set_default("all"));
    ASSERT_TRUE(l.is_extended());
    ASSERT_TRUE(l.is_default());
    ASSERT_EQ(1, l.size());
    ASSERT_EQ(0, l.values());
    ASSERT_STREQ("all", l.get_default());
    ASSERT_STREQ("all", l.get_value());
    ASSERT_TRUE(l.set_value("1,2,  3,  4"));
    ASSERT_STREQ("1,2,3,4", l.get_value());
    ASSERT_FALSE(l.is_default());
    ASSERT_FALSE(l.is_extended());
    ASSERT_EQ(4, l.size());
    ASSERT_EQ(1, l.values()[0]);
    ASSERT_EQ(2, l.values()[1]);
    ASSERT_EQ(3, l.values()[2]);
    ASSERT_EQ(4, l.values()[3]);
    ASSERT_FALSE(l.set_default("foo"));
    ASSERT_TRUE(l.set_value("2"));
    ASSERT_STREQ("2", l.get_value());
    ASSERT_EQ(1, l.size());
    ASSERT_EQ(2, l.values()[0]);

    // Comparison.
    {
        IntListExt l1, l2;
        ASSERT_TRUE(l1.set_value("1,2,3,4"));
        ASSERT_TRUE(l2.set_value("1,2,3,4"));
        ASSERT_TRUE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
    }
}
