/*
 * Copyright (c) 2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstring>

#include <gtest/gtest.h>

#include "settings/oskar_option_parser.h"

// Create some test settings.
static const char* test_settings_xml = ""
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
        "<root version=\"2.12.0\">"
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
        "        <s k=\"list_key\"><label>A list</label>"
        "            <type name=\"OptionList\" default=\"XY\">XY,X,Y</type>"
        "            <desc>This is a list.</desc>"
        "        </s>"
        "    </s>"
        "</root>";


TEST(OptionParser, test_options)
{
    // Create a test parser.
    oskar::OptionParser opt("test", "0.1");
    opt.add_settings_options();
    opt.set_description("A test option parser");
    opt.set_settings(test_settings_xml);
    opt.set_title("test");
    opt.set_version("0.1.2");
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");
    opt.add_flag("-nst", "Number of stations", 1, "", true);
    opt.add_flag("-n", "Number of iterations", 1, "1", false);
    opt.add_flag("-o", "Output name", 1, "out.dat", false, "--output");
    opt.add_required("Test files...");
    opt.add_example("test -q -nst 123 -n 2 settings.ini file1 file2");

    // Print usage.
    opt.print_usage();

    // Happy path.
    {
        const char* my_argv_const[] = {
                "test", "-q", "-nst", "123", "-n", "2",
                "settings.ini", "file1", "file2"
        };
        const int my_argc = sizeof(my_argv_const) / sizeof(const char*);
        char** my_argv = (char**) calloc(my_argc, sizeof(char*));
        for (int i = 0; i < my_argc; ++i)
        {
            const size_t len = strlen(my_argv_const[i]);
            my_argv[i] = (char*) calloc(1 + len, sizeof(char));
            memcpy(my_argv[i], my_argv_const[i], len);
        }

        // Check options.
        ASSERT_TRUE(opt.check_options(my_argc, my_argv));

        // Check values.
        int num_in_files = 0;
        const char* const* in_files = opt.get_input_files(1, &num_in_files);
        ASSERT_EQ(3, num_in_files);
        ASSERT_STREQ("settings.ini", in_files[0]);
        ASSERT_STREQ("file1", in_files[1]);
        ASSERT_STREQ("file2", in_files[2]);
        ASSERT_EQ(123, opt.get_int("-nst"));
        ASSERT_EQ(2, opt.get_int("-n"));
        ASSERT_STREQ("2", opt.get_string("-n"));
        ASSERT_STREQ("out.dat", opt.get_string("-o"));
        ASSERT_STREQ("out.dat", opt.get_string("--output"));
        ASSERT_DOUBLE_EQ(2.0, opt.get_double("-n"));
        ASSERT_TRUE(opt.is_set("-q"));
        ASSERT_STREQ("settings.ini", opt.get_arg(0));

        // Clean up.
        for (int i = 0; i < my_argc; ++i)
        {
            free(my_argv[i]);
        }
        free(my_argv);
    }
}


TEST(OptionParser, print)
{
    oskar::OptionParser opt("test", "0.1");
    opt.add_settings_options();
    opt.set_description("A test option parser");
    opt.set_settings(test_settings_xml);
    opt.set_title("test");
    opt.set_version("0.1.2");
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");

    // Print help.
    {
        const char* my_argv_const[] = {"test", "--help"};
        const int my_argc = sizeof(my_argv_const) / sizeof(const char*);
        char** my_argv = (char**) calloc(my_argc, sizeof(char*));
        for (int i = 0; i < my_argc; ++i)
        {
            const size_t len = strlen(my_argv_const[i]);
            my_argv[i] = (char*) calloc(1 + len, sizeof(char));
            memcpy(my_argv[i], my_argv_const[i], len);
        }

        // Check options.
        ASSERT_FALSE(opt.check_options(my_argc, my_argv));

        // Clean up.
        for (int i = 0; i < my_argc; ++i)
        {
            free(my_argv[i]);
        }
        free(my_argv);
    }

    // Print version.
    {
        const char* my_argv_const[] = {"test", "--version"};
        const int my_argc = sizeof(my_argv_const) / sizeof(const char*);
        char** my_argv = (char**) calloc(my_argc, sizeof(char*));
        for (int i = 0; i < my_argc; ++i)
        {
            const size_t len = strlen(my_argv_const[i]);
            my_argv[i] = (char*) calloc(1 + len, sizeof(char));
            memcpy(my_argv[i], my_argv_const[i], len);
        }

        // Check options.
        ASSERT_FALSE(opt.check_options(my_argc, my_argv));

        // Clean up.
        for (int i = 0; i < my_argc; ++i)
        {
            free(my_argv[i]);
        }
        free(my_argv);
    }

    // Print settings.
    {
        const char* my_argv_const[] = {"test", "--settings"};
        const int my_argc = sizeof(my_argv_const) / sizeof(const char*);
        char** my_argv = (char**) calloc(my_argc, sizeof(char*));
        for (int i = 0; i < my_argc; ++i)
        {
            const size_t len = strlen(my_argv_const[i]);
            my_argv[i] = (char*) calloc(1 + len, sizeof(char));
            memcpy(my_argv[i], my_argv_const[i], len);
        }

        // Check options.
        ASSERT_FALSE(opt.check_options(my_argc, my_argv));

        // Clean up.
        for (int i = 0; i < my_argc; ++i)
        {
            free(my_argv[i]);
        }
        free(my_argv);
    }
}


TEST(OptionParser, not_enough_arguments)
{
    oskar::OptionParser opt("test", "0.1");
    opt.add_settings_options();
    opt.set_description("A test option parser");
    opt.set_settings(test_settings_xml);
    opt.set_title("test");
    opt.set_version("0.1.2");
    opt.add_flag("-q", "Suppress printing.", false, "--quiet");

    const char* my_argv_const[] = {"test"};
    const int my_argc = sizeof(my_argv_const) / sizeof(const char*);
    char** my_argv = (char**) calloc(my_argc, sizeof(char*));
    for (int i = 0; i < my_argc; ++i)
    {
        const size_t len = strlen(my_argv_const[i]);
        my_argv[i] = (char*) calloc(1 + len, sizeof(char));
        memcpy(my_argv[i], my_argv_const[i], len);
    }

    // Check options.
    ASSERT_FALSE(opt.check_options(my_argc, my_argv));

    // Clean up.
    for (int i = 0; i < my_argc; ++i)
    {
        free(my_argv[i]);
    }
    free(my_argv);
}
