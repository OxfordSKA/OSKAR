/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, IntList)
{
    IntList l;
    ASSERT_TRUE(l.set_default("1,2,3, 4,5"));
    ASSERT_STREQ("1,2,3,4,5", l.get_default());
    ASSERT_STREQ("1,2,3,4,5", l.get_value());
    ASSERT_EQ(5, l.size());
    for (int i = 0; i < l.size(); ++i) {
        ASSERT_EQ(i+1, l.values()[i]);
    }
    ASSERT_TRUE(l.is_default());
    ASSERT_TRUE(l.set_value("9, 10, 11"));
    ASSERT_FALSE(l.is_default());
    ASSERT_EQ(3, l.size());
    ASSERT_EQ(9, l.values()[0]);
    ASSERT_EQ(10, l.values()[1]);
    ASSERT_EQ(11, l.values()[2]);
    ASSERT_STREQ("9,10,11", l.get_value());

    // Comparison.
    {
        IntList l1, l2;
        ASSERT_TRUE(l1.set_value("9, 10, 11"));
        ASSERT_TRUE(l2.set_value("9, 10, 11"));
        ASSERT_TRUE(l1 == l2);
        ASSERT_TRUE(l2.set_value("9, 10"));
        ASSERT_FALSE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
    }
}
