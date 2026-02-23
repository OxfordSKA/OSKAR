/*
 * Copyright (c) 2015-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "settings/oskar_SettingsKey.h"


TEST(SettingsKey, test1)
{
    oskar::SettingsKey k("a/b/c/d");
    ASSERT_EQ(3, k.depth());
    ASSERT_STREQ("a", k[0]);
    ASSERT_STREQ("b", k[1]);
    ASSERT_STREQ("c", k[2]);
    ASSERT_STREQ("d", k[3]);

    ASSERT_STREQ("d", k.back());
    ASSERT_STREQ("a/b/c/d", k);
}


TEST(SettingsKey, different_separator)
{
    oskar::SettingsKey k("a.b.c.d", '.');
    ASSERT_EQ('.', k.separator());
    ASSERT_EQ(3, k.depth());
    ASSERT_STREQ("a", k[0]);
    ASSERT_STREQ("b", k[1]);
    ASSERT_STREQ("c", k[2]);
    ASSERT_STREQ("d", k[3]);

    ASSERT_STREQ("d", k.back());
    ASSERT_STREQ("a.b.c.d", k);

    k.set_separator('-');
    k.from_string("e-f-g-h", k.separator());
    ASSERT_EQ(3, k.depth());
    ASSERT_STREQ("e", k[0]);
    ASSERT_STREQ("f", k[1]);
    ASSERT_STREQ("g", k[2]);
    ASSERT_STREQ("h", k[3]);

    ASSERT_STREQ("h", k.back());
    ASSERT_STREQ("e-f-g-h", k);
}
