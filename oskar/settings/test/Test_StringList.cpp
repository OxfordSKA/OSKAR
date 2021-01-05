/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;
using namespace std;

TEST(settings_types, StringList)
{
    StringList l;
    ASSERT_TRUE(l.init(""));
    ASSERT_TRUE(l.set_default("a,b,c"));
    ASSERT_EQ(3, l.size());
    ASSERT_STREQ("a", l.values()[0]);
    ASSERT_STREQ("b", l.values()[1]);
    ASSERT_STREQ("c", l.values()[2]);
    ASSERT_STREQ("a,b,c", l.get_default());
    ASSERT_TRUE(l.is_default());

    ASSERT_TRUE(l.set_value("p, q"));
    ASSERT_EQ(2, l.size());
    ASSERT_STREQ("p", l.values()[0]);
    ASSERT_STREQ("q", l.values()[1]);
    ASSERT_STREQ("p,q", l.get_value());
    ASSERT_FALSE(l.is_default());

    // Comparison.
    {
        StringList l1, l2;
        ASSERT_TRUE(l1.set_value("a, b"));
        ASSERT_TRUE(l2.set_value("a, b"));
        ASSERT_EQ(2, l1.size());
        ASSERT_EQ(2, l2.size());
        ASSERT_TRUE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
        ASSERT_TRUE(l2.set_value("b, c"));
        ASSERT_EQ(2, l2.size());
        ASSERT_FALSE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
    }
}
