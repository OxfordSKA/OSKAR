/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>
#include "settings/oskar_SettingsItem.h"
#include "settings/oskar_SettingsValue.h"
#include "settings/oskar_SettingsDependencyGroup.h"

using namespace oskar;
using namespace std;

TEST(SettingsItem, test)
{
    {
        SettingsItem i;
        ASSERT_EQ(SettingsItem::INVALID, i.item_type());
    }
    {
        SettingsItem i("key");
        ASSERT_EQ(SettingsItem::LABEL, i.item_type());
    }
    {
        SettingsItem i("key", "label", "description",
                       "IntRange", "5", "5, 10");
        ASSERT_EQ(SettingsItem::SETTING, i.item_type());
        ASSERT_EQ(SettingsValue::INT_RANGE, i.settings_value().type());
        ASSERT_TRUE(i.set_value("7"));
        ASSERT_STREQ("7", i.value());
        // TODO(BM) test ability to check the range from this interface.
        // TODO(BM) test ability to trap failed initialisation of types.
    }
}

TEST(SettingsItem, simple_deps)
{
    {
        SettingsItem i("key", "", "", "Double");
        ASSERT_EQ(SettingsItem::SETTING, i.item_type());
        ASSERT_EQ(SettingsValue::DOUBLE, i.settings_value().type());
        ASSERT_STREQ("0.0", i.value());
        ASSERT_TRUE(i.add_dependency("another_key", "true", "EQ"));
        ASSERT_EQ(1, i.num_dependencies());
    }
    {
        SettingsItem i("keyA");
        i.add_dependency("keyB", "true");
        i.add_dependency("keyC", "2", "NE");
        ASSERT_EQ(2, i.num_dependencies());
        ASSERT_EQ(1, i.num_dependency_groups());
    }
}

TEST(SettingsItem, nested_deps)
{
    SettingsItem i("key");
    i.begin_dependency_group("AND");
    i.begin_dependency_group("OR");
    i.end_dependency_group();
    i.end_dependency_group();
    ASSERT_EQ(2, i.num_dependency_groups());

    const SettingsDependencyGroup* deps = i.dependency_tree();
    ASSERT_EQ(1, deps->num_children());
    ASSERT_EQ(SettingsDependencyGroup::AND, deps->group_logic());
    ASSERT_EQ(0, deps->get_child(0)->num_children());
    ASSERT_EQ(SettingsDependencyGroup::OR, deps->get_child(0)->group_logic());
    //ASSERT_EQ(1, deps->get_child(0)->num_children());
}
