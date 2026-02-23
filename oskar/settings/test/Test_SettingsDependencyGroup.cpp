/*
 * Copyright (c) 2015-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "settings/oskar_SettingsDependencyGroup.h"

using namespace oskar;

TEST(SettingsDependencyGroup, test1)
{
    SettingsDependencyGroup* g = new SettingsDependencyGroup("OR");
    ASSERT_EQ(SettingsDependencyGroup::OR, g->group_logic());
    g->add_dependency(SettingsDependency("key1", "value1", "EQ"));
    g->add_dependency(SettingsDependency("key2", "value2", "NE"));
    ASSERT_EQ(2, g->num_dependencies());
    ASSERT_EQ(0, g->num_children());
    ASSERT_EQ(SettingsDependencyGroup::OR, g->string_to_group_logic("OR"));
    ASSERT_EQ(SettingsDependencyGroup::AND, g->string_to_group_logic("AND"));
    ASSERT_EQ(SettingsDependencyGroup::AND, g->string_to_group_logic(""));
    ASSERT_EQ(SettingsDependencyGroup::UNDEF, g->string_to_group_logic("um"));

    ASSERT_STREQ("key1", g->get_dependency(0)->key());
    ASSERT_STREQ("value1", g->get_dependency(0)->value());
    ASSERT_EQ(SettingsDependency::EQ, g->get_dependency(0)->logic());

    ASSERT_STREQ("key2", g->get_dependency(1)->key());
    ASSERT_STREQ("value2", g->get_dependency(1)->value());
    ASSERT_EQ(SettingsDependency::NE, g->get_dependency(1)->logic());

    SettingsDependencyGroup* group = g->add_child("AND");
    ASSERT_EQ(1, g->num_children());

    ASSERT_EQ(SettingsDependencyGroup::AND, group->group_logic());
    group->add_dependency("key3", "2.0", "GT");
    ASSERT_EQ(1, group->num_dependencies());
    delete g;
}
