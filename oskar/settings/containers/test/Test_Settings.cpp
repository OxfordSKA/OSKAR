/*
 * Copyright (c) 2015, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the OSKAR package.
 * Contact: oskar at oerc.ox.ac.uk
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

#include "oskar_SettingsTree.hpp"
#include "oskar_SettingsDeclareXml.hpp"
#include "oskar_settings_utility_string.hpp"
#include <string>

using namespace oskar;
using namespace std;

TEST(SettingsTree, test)
{
    SettingsTree s;
    ASSERT_TRUE(s.add_setting("sky",
                                 "Sky settings",
                                 "description of sky settings"));
    ASSERT_TRUE(s.add_setting("sky/file",
                                 "Sky file",
                                 "description of sky file",
                                 "InputFileList"));
    ASSERT_TRUE(s.add_setting("sky/generator/type",
                                 "generator type",
                                 "description of generator",
                                 "OptionList",
                                 "b",
                                 "a,b,c"));
    ASSERT_TRUE(s.add_setting("telescope/directory",
                                 "telescope dir",
                                 "description",
                                 "InputDirectory",
                                 "", "", true));

    ASSERT_EQ(6, s.num_items());
    ASSERT_EQ(3, s.num_settings());

    ASSERT_EQ("b", s.value("sky/generator/type")->get_value());
    ASSERT_TRUE(s.item("telescope/directory")->is_required());

    ASSERT_TRUE(s.set_value("sky/file", "sky.osm"));
    ASSERT_EQ("sky.osm", s["sky/file"]->get_value());
    ASSERT_FALSE(s.contains("a/b/c"));
    s.print();

    s.clear();
    ASSERT_EQ(0, s.num_items());
    ASSERT_EQ(0, s.num_settings());

    s.begin_group("a");
    {
        s.add_setting("b","","", "Int");
        s.begin_group("c");
        {
            s.add_setting("d", "", "", "Double");
        }
        s.end_group();
    }
    s.end_group();
    s.print();

    s.clear();
    ASSERT_FALSE(s.add_setting("key", "", "", "Int", "hello"));
    ASSERT_EQ(0, s.num_items());
    ASSERT_EQ(0, s.num_settings());

    s.clear();
}

TEST(SettingsTree, xml)
{
    const char* temp = ""
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
            "<root version=\"2.6.1\">"
            "    <s k=\"group1\">"
            "        <label>Label for Group 1</label>"
            "        <s key=\"key1\"><label>A bool</label>"
            "            <type name=\"bool\" default=\"false\"/>"
            "            <desc>"
            "                This is a bool. The description "
            "                can span multiple lines like this."
            "            </desc>"
            "        </s>"
            "        <s key=\"key2\">"
            "            <label>An int</label>"
            "            <type name=\"int\" default=\"5000\"/>"
            "        </s>"
            "        <s key=\"key3\">"
            "            <type name=\"bool\" default=\"true\"/>"
            "            <s key=\"key4\">"
            "                <type name=\"int\" default=\"10\"/>"
            "            </s>"
            "        </s>"
            "        <s k=\"input_file_key\"><label>Input file</label>"
            "            <type name=\"InputFile\" default=\"\"/>"
            "            <desc>Pathname to input file.</desc>"
            "        </s>"
            "        <s k=\"unsigned_double_key\">"
            "            <label>An unsigned double</label>"
            "            <type name=\"UnsignedDouble\" default=\"0.0\"/>"
            "            <desc>This is an unsigned double.</desc>"
            "        </s>"
            "        <s k=\"list_key\"><label>A list</label>"
            "            <type name=\"OptionList\" default=\"XY\">XY,X,Y</type>"
            "            <desc>This is a list.</desc>"
            "        </s>"
            "        <s k=\"uint_key\">"
            "            <label>An unsigned int</label>"
            "            <type name=\"uint\" default=\"0\"/>"
            "            <desc>This is an unsigned int.</desc>"
            "        </s>"
            "    </s>"
            "    <s key=\"group2\">"
            "        <!-- Setting with simple dependency -->"
            "        <s key=\"simple_deps\">"
            "            <type name=\"string\" default=\"\"/>"
            "            <d k=\"group1/key1\" v=\"true\"/>"
            "            <d k=\"group1/key2\" v=\"5\"/>"
            "        </s>"
            "        <!-- Setting with more complicated dependency logic -->"
            "        <s key=\"nested_deps\">"
            "            <type name=\"string\" default=\"\"/>"
            "            <logic group=\"OR\">"
            "                <d k=\"group1/key1\" c=\"NE\" v=\"true\"/>"
            "                <l g=\"OR\">"
            "                    <d k=\"group1/key2\" c=\"GT\" v=\"200\"/>"
            "                    <d k=\"group1/key3\" c=\"EQ\" v=\"false\"/>"
            "                </l>"
            "                <d k=\"group1/key3/key4\" c=\"EQ\" v=\"2\"/>"
            "            </l>"
            "        </s>"
            "    </s>"
            "</root>";

    // Create the settings tree and declare settings using XML.
    SettingsTree s;
    ASSERT_TRUE(settings_declare_xml(&s, temp));

    // Get a value out of the tree and check it.
    int status = 0;
    ASSERT_EQ(5000, s.to_int("group1/key2", &status));
    ASSERT_EQ(0, status);

    // Print the settings tree.
    s.print();
}

TEST(SettingsTree, simple_deps)
{
    SettingsTree s;
    s.add_setting("keyA", "", "", "Bool", "false");

    s.add_setting("keyB");
    s.add_dependency("keyA", "true", "EQ");
    s.add_setting("keyC");
    s.add_dependency("KeyA", "true", "EQ");

    ASSERT_EQ(0, s.item("keyA")->num_dependencies());
    ASSERT_EQ(1, s.item("keyB")->num_dependencies());

    EXPECT_FALSE(s.dependencies_satisfied("keyB"));

    ASSERT_TRUE(s.set_value("keyA", "true"));

    ASSERT_EQ("true", s.value("keyA")->get_value());
    ASSERT_EQ("true", s["keyA"]->get_value());

    ASSERT_TRUE(s.dependencies_satisfied("KEYB"));
}

TEST(SettingsTree, nested_deps)
{
    SettingsTree s;
    s.add_setting("keyA1", "", "", "Bool", "false");
    s.add_setting("keyA2", "", "", "Int", "2");

    s.add_setting("keyB");
    s.begin_dependency_group("AND");
    s.add_dependency("KeyA1", "true", "EQ");
    s.add_dependency("KeyA2", "3", "GT");
    s.end_dependency_group();

    ASSERT_EQ(0, s.item("keyA1")->num_dependencies());
    ASSERT_EQ(0, s.item("keyA2")->num_dependencies());
    ASSERT_EQ(2, s.item("keyB")->num_dependencies());

    ASSERT_FALSE(s.dependencies_satisfied("keyB"));
    ASSERT_TRUE(s.set_value("keyA1", "true"));
    ASSERT_TRUE(s.set_value("keyA2", "5"));
    ASSERT_TRUE(s.dependencies_satisfied("keyB"));
}

