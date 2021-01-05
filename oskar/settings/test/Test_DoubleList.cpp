/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, DoubleList)
{
    DoubleList l;
    {
        ASSERT_TRUE(l.set_default("0.0,1.1,2.2,3.3,4.4,5.5"));
        ASSERT_TRUE(l.is_default());
        ASSERT_EQ(6, l.size());
        for (int i = 0; i < l.size(); ++i) {
            ASSERT_EQ(double(i)+double(i)/10., l.values()[i]);
        }
        ASSERT_STREQ("0.0,1.1,2.2,3.3,4.4,5.5", l.get_value());
    }
    {
        ASSERT_TRUE(l.set_value("0.01234567891011,  666.6"));
        ASSERT_FALSE(l.is_default());
        ASSERT_EQ(2, l.size());
        ASSERT_DOUBLE_EQ(0.01234567891011, l.values()[0]);
        ASSERT_EQ(666.6, l.values()[1]);
        // FIXME(BM) string printing to not enough decimal places!
        EXPECT_STREQ("0.01234567891011,666.6", l.get_value());
    }

    // Comparison.
    {
        DoubleList l1, l2;
        ASSERT_TRUE(l1.set_value("1.1, 2.2"));
        ASSERT_TRUE(l2.set_value("1.1, 2.2"));
        ASSERT_TRUE(l1 == l2);
        ASSERT_FALSE(l1 > l2);
    }
}

