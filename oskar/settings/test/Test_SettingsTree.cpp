/*
 * Copyright (c) 2015-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "settings/oskar_SettingsDeclareXml.h"
#include "settings/oskar_SettingsFileHandlerIni.h"
#include "settings/oskar_SettingsItem.h"
#include "settings/oskar_SettingsNode.h"
#include "settings/oskar_SettingsTree.h"
#include "settings/oskar_settings_utility_string.h"


// Create some test settings.
static const char* test_settings_xml = ""
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
        "            <type name=\"OptionList\" default=\"A\">A,B,C</type>"
        "            <desc>This is a list.</desc>"
        "        </s>"
        "        <s k=\"uint_key\" required=\"true\">"
        "            <label>An unsigned int</label>"
        "            <type name=\"uint\"/>"
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


TEST(SettingsTree, basic_tests)
{
    oskar::SettingsTree s;
    ASSERT_TRUE(
            s.add_setting("sky", "Sky settings", "description of sky settings")
    );
    ASSERT_TRUE(
            s.add_setting(
                    "sky/file", "Sky file", "description of sky file",
                    "InputFileList"
            )
    );
    ASSERT_TRUE(
            s.add_setting(
                    "sky/generator/type", "generator type",
                    "description of generator",
                    "OptionList", "b", "a,b,c"
            )
    );
    ASSERT_TRUE(
            s.add_setting(
                    "telescope/directory", "telescope dir", "description",
                    "InputDirectory", "", "", true
            )
    );
    ASSERT_EQ(6, s.num_items());
    ASSERT_EQ(3, s.num_settings());

    // Read default parameters.
    {
        int status = 0;
        ASSERT_STREQ("b", s.to_string("sky/generator/type", &status));
        ASSERT_EQ(0, status);
        ASSERT_TRUE(s.item("telescope/directory")->is_required());
    }

    // Set a value and read it back.
    {
        ASSERT_TRUE(s.set_value("sky/file", "sky.osm"));
        ASSERT_STREQ("sky.osm", s["sky/file"]);
    }

    // Check that the tree does not contain the specified key.
    ASSERT_FALSE(s.contains("a/b/c"));

    // Print the current tree.
    s.print();

    // Clear the whole tree and check that it is empty.
    s.clear();
    ASSERT_EQ(0, s.num_items());
    ASSERT_EQ(0, s.num_settings());

    // Add new settings and print the tree.
    s.begin_group("a");
    {
        s.add_setting("b", "", "", "Int");
        s.begin_group("c");
        {
            s.add_setting("d", "", "", "Double");
        }
        s.end_group();
    }
    s.end_group();
    s.print();

    // Clear the whole tree and try to add a setting with a broken default.
    s.clear();
    ASSERT_FALSE(s.add_setting("key", "", "", "Int", "hello"));
    ASSERT_EQ(0, s.num_items());
    ASSERT_EQ(0, s.num_settings());

    // Try to find a setting that doesn't exist.
    ASSERT_FALSE(s.item("a_key_that_does_not_exist"));
    ASSERT_FALSE(s.dependencies_satisfied("another_non_existent_key"));
}


TEST(SettingsTree, basic_deps)
{
    oskar::SettingsTree s;
    s.add_setting("g1", "group 1", "description of group 1");
    s.begin_group("g1");
    {
        s.add_setting("b", "setting b", "", "Int");
        s.add_setting(
                "c", "OptionList", "description for c",
                "OptionList", "", "opt1, opt2, opt3, opt4", true
        );
        s.add_setting(
                "d", "setting d", "description for d",
                "InputFile", "", "", true
        );
        s.begin_dependency_group("OR");
        s.add_dependency("g1/b", "3", "GE");
        s.add_dependency("g1/c", "opt4", "EQ");
        s.end_dependency_group();
        s.end_group();
    }
    s.begin_group("g2");
    {
        s.add_setting(
                "DoubleRangeExt", "DoubleRangeExt", "",
                "DoubleRangeExt", "min", "-DBL_MIN,DBL_MAX,min"
        );
        s.add_setting(
                "IntRangeExt", "IntRangeExt", "",
                "IntRangeExt", "5", "-1,INT_MAX,min"
        );
        s.end_group();
    }
    s.begin_group("g3");
    {
        s.begin_group("g4");
        {
            s.add_setting(
                    "IntListExt", "IntListExt", "",
                    "IntListExt", "1,2,3", "all"
            );
            s.end_group();
        }
        s.add_setting("Double1", "Double1", "", "Double", "-10.0");
        s.add_setting("Double2", "Double2", "", "Double", "100.0e3");
        s.end_group();
    }

    int status = 0;
    ASSERT_EQ(0, s.to_int("g1/b", &status));
    ASSERT_EQ(0, status);
    ASSERT_TRUE(s.dependencies_satisfied("g1"));
    ASSERT_TRUE(s.dependencies_satisfied("g1/b"));
    ASSERT_FALSE(s.dependencies_satisfied("g1/d"));
}


TEST(SettingsTree, xml)
{
    // Create the settings tree and declare settings using XML.
    oskar::SettingsTree s;
    ASSERT_TRUE(oskar_settings_declare_xml(&s, test_settings_xml));

    // Get a value out of the tree and check it.
    int status = 0;
    ASSERT_EQ(5000, s.to_int("group1/key2", &status));
    ASSERT_EQ(0, status);

    // Get an item from the tree and check the label and description.
    const oskar::SettingsItem* item = s.item("group1/list_key");
    ASSERT_STREQ("A list", item->label());
    ASSERT_STREQ("This is a list.", item->description());

    // Check a node.
    const oskar::SettingsNode* root_node = s.root_node();
    ASSERT_TRUE(root_node);
    const oskar::SettingsNode* group1 = root_node->child("group1");
    ASSERT_TRUE(group1);
    const oskar::SettingsNode* key2 = group1->child("group1/key2");
    ASSERT_TRUE(key2);
    ASSERT_EQ(group1, key2->parent());
    const oskar::SettingsNode* key3 = group1->child("group1/key3");
    ASSERT_TRUE(key3);
    ASSERT_EQ(group1, key3->parent());
    EXPECT_EQ(1, key2->child_number());
    EXPECT_EQ(2, key3->child_number());

    // Print the settings tree.
    s.print();
}


TEST(SettingsTree, simple_deps)
{
    oskar::SettingsTree s;
    s.add_setting("keyA", "", "", "Bool", "false");

    s.add_setting("keyB");
    s.add_dependency("keyA", "true", "EQ");
    s.add_setting("keyC");
    s.add_dependency("KeyA", "true", "EQ");

    ASSERT_EQ(0, s.item("keyA")->num_dependencies());
    ASSERT_EQ(1, s.item("keyB")->num_dependencies());

    EXPECT_FALSE(s.dependencies_satisfied("keyB"));

    ASSERT_TRUE(s.set_value("keyA", "true"));

    int status = 0;
    ASSERT_STREQ("true", s.to_string("keyA", &status));
    ASSERT_EQ(0, status);
    ASSERT_STREQ("true", s["keyA"]);

    ASSERT_TRUE(s.dependencies_satisfied("KEYB"));
}


TEST(SettingsTree, nested_deps)
{
    oskar::SettingsTree s;
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


TEST(SettingsTree, ini_file_basic)
{
    const char* file_name = "temp_test_settings.ini";

    // Create a settings file based on the test XML tree.
    {
        oskar::SettingsTree s;
        ASSERT_TRUE(oskar_settings_declare_xml(&s, test_settings_xml));
        oskar::SettingsFileHandlerIni* handler =
                new oskar::SettingsFileHandlerIni("test", "0.1");
        handler->set_write_defaults(true);
        ASSERT_FALSE(s.is_modified());
        ASSERT_TRUE(s.is_critical("group1"));
        s.begin_group("group1");
        {
            std::string group_prefix = std::string("group1") + s.separator();
            ASSERT_TRUE(s.set_value("key1", "true"));
            ASSERT_STREQ(group_prefix.c_str(), s.group_prefix());
            ASSERT_TRUE(s.is_modified());
            ASSERT_TRUE(s.set_value("key2", "5"));
            ASSERT_TRUE(s.is_critical("uint_key"));
            s.set_file_handler(handler);
            s.set_file_name(file_name);
            s.save();
            ASSERT_FALSE(s.is_modified());
            int status = 0;
            ASSERT_TRUE(s.set_value("uint_key", "42"));
            ASSERT_FALSE(s.is_critical("uint_key"));
            ASSERT_TRUE(s.set_value("list_key", "B"));
            ASSERT_TRUE(s.set_default("list_key"));
            ASSERT_STREQ("A", s.to_string("list_key", &status));
            ASSERT_EQ(0, status);
            ASSERT_TRUE(s.set_value("list_key", "B"));
            ASSERT_STREQ("B", s.to_string("list_key", &status));
            ASSERT_EQ(0, status);
            s.end_group();
        }
        ASSERT_TRUE(s.set_value("group2/simple_deps", "hello"));
        ASSERT_FALSE(s.is_modified());
    }

    // Load back the settings file.
    {
        oskar::SettingsTree s;
        ASSERT_TRUE(oskar_settings_declare_xml(&s, test_settings_xml));
        oskar::SettingsFileHandlerIni* handler =
                new oskar::SettingsFileHandlerIni("test", "0.1");
        s.set_file_handler(handler);
        s.load(file_name);
        ASSERT_EQ(0, s.num_failed_keys());
        ASSERT_FALSE(s.failed_key(0));
        ASSERT_FALSE(s.failed_key_value(0));
        char* app_name = s.file_handler()->read(file_name, "app");
        ASSERT_STREQ("test", app_name);
        free(app_name);
        int status = 0;
        ASSERT_EQ(1, s.to_int("group1/key1", &status));
        ASSERT_EQ(0, status);
        ASSERT_EQ(5, s.to_int("group1/key2", &status));
        ASSERT_EQ(0, status);
        ASSERT_STREQ("B", s.to_string("group1/list_key", &status));
        ASSERT_EQ(0, status);
        ASSERT_STREQ("hello", s.to_string("group2/simple_deps", &status));
        ASSERT_EQ(0, status);
    }
    remove(file_name);
}
