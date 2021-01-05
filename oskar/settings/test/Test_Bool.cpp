/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, Bool)
{
    Bool b;
    ASSERT_TRUE(b.set_default("false"));
    ASSERT_STREQ("false", b.get_default());
    ASSERT_STREQ("false", b.get_value());
    ASSERT_TRUE(b.is_default());
    ASSERT_TRUE(b.set_value("true"));
    ASSERT_STREQ("true", b.get_value());
    ASSERT_FALSE(b.is_default());
    ASSERT_TRUE(b.value());

    // Comparison.
    {
        Bool b1, b2;
        ASSERT_TRUE(b1.set_value("true"));
        ASSERT_TRUE(b2.set_value("true"));
        ASSERT_FALSE(b1 > b2);
        ASSERT_TRUE(b1 == b2);
    }
}

