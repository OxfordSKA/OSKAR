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
#include "settings/oskar_SettingsDependencyGroup.h"

using namespace oskar;
using namespace std;

TEST(SettingsDependencyGroup, test1)
{
    SettingsDependencyGroup* g = new SettingsDependencyGroup("OR");
    ASSERT_EQ(SettingsDependencyGroup::OR, g->group_logic());
    g->add_dependency(SettingsDependency("key1", "value1", "EQ"));
    g->add_dependency(SettingsDependency("key2", "value2", "NE"));
    ASSERT_EQ(2, g->num_dependencies());
    ASSERT_EQ(0, g->num_children());

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

