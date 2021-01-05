/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, OptionList)
{
    OptionList l;
    ASSERT_TRUE(l.init("a,    b,"
                    ""
                    "c"));
    ASSERT_EQ(3, l.size());
    ASSERT_STREQ("a", l.get_value());
    ASSERT_TRUE(l.is_default());
    ASSERT_FALSE(l.set_default("z"));
    ASSERT_STREQ("a", l.get_default());
    ASSERT_TRUE(l.set_default("b"));
    ASSERT_STREQ("b", l.get_default());
    ASSERT_TRUE(l.set_value("a"));
    ASSERT_FALSE(l.is_default());
    ASSERT_STREQ("a", l.get_value());

    // Comparison.
    {
        OptionList l1, l2;
        ASSERT_TRUE(l1.init("a1,b,c"));
        ASSERT_TRUE(l2.init("a2,b,c"));
        ASSERT_TRUE(l1.set_value("b"));
        ASSERT_TRUE(l2.set_value("b"));
        ASSERT_TRUE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
        ASSERT_STREQ("a1", l1.option(0));
        ASSERT_STREQ("b", l1.option(1));
        ASSERT_STREQ("a2", l2.option(0));
        ASSERT_STREQ("b", l2.option(1));
    }
}
