/*
 * Copyright (c) 2015-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "settings/oskar_SettingsDependency.h"

TEST(SettingsDependency, test1)
{
    typedef oskar::SettingsDependency SD;
    oskar::SettingsDependency dependency("a/b/c", "2.0", "EQ");
    ASSERT_STREQ("a/b/c", dependency.key());
    ASSERT_STREQ("2.0", dependency.value());
    ASSERT_EQ(SD::EQ, dependency.logic());
    ASSERT_STREQ("EQ", dependency.logic_string());
    ASSERT_TRUE(dependency.is_valid());

    ASSERT_STREQ("EQ", SD::logic_to_string(SD::EQ));
    ASSERT_STREQ("NE", SD::logic_to_string(SD::NE));
    ASSERT_STREQ("GT", SD::logic_to_string(SD::GT));
    ASSERT_STREQ("GE", SD::logic_to_string(SD::GE));
    ASSERT_STREQ("LT", SD::logic_to_string(SD::LT));
    ASSERT_STREQ("LE", SD::logic_to_string(SD::LE));
    ASSERT_STREQ("", SD::logic_to_string(SD::UNDEF));
    ASSERT_EQ(SD::EQ, SD::string_to_logic("EQ"));
    ASSERT_EQ(SD::NE, SD::string_to_logic("NE"));
    ASSERT_EQ(SD::GT, SD::string_to_logic("GT"));
    ASSERT_EQ(SD::GE, SD::string_to_logic("GE"));
    ASSERT_EQ(SD::LT, SD::string_to_logic("LT"));
    ASSERT_EQ(SD::LE, SD::string_to_logic("LE"));
}
