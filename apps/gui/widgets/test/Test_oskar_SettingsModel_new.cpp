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
#include <oskar_SettingsFileHandlerQSettings.hpp>
#include <oskar_SettingsModel_new.hpp>
#include <oskar_SettingsDeclareXml.hpp>
#include <oskar_SettingsView.h>
#include <oskar_SettingsDelegate_new.hpp>
#include <QtGui/QApplication>
#include <QtGui/QWidget>
#include <QtGui/QVBoxLayout>

#include "apps/xml/oskar_xml_all.h"

using namespace std;
using namespace oskar;

TEST(test_SettingsModel_new, test1)
{
    int argc = 1;
    char** argv = (char**)malloc(argc * sizeof(char*));
    for (int i = 0; i < argc; ++i)
        argv[i] = (char*)malloc(255 * sizeof(char));
    QApplication app(argc, argv);
    QWidget widget;

    SettingsTree s;
    s.add_setting("g1", "group 1", "description of group 1");
    s.begin_group("g1");
    {
        s.add_setting("b", "setting b", "", "Int");
        s.add_setting("c", "OptionList", "description for c", "OptionList",
                         "", "opt1, opt2, opt3, opt4", true);
        s.add_setting("d", "setting d", "description or d", "InputFile",
                         "", "", true);
        s.begin_dependency_group("OR");
        s.add_dependency("g1/b", "3", "GE");
        s.add_dependency("g1/c", "opt4", "EQ");
        s.end_dependency_group();
        s.end_group();
    }
    s.begin_group("g2");
    {
        s.add_setting("DoubleRangeExt", "DoubleRangeExt", "",
                         "DoubleRangeExt", "min", "-DBL_MIN,DBL_MAX,min");
        s.add_setting("IntRangeExt", "IntRangeExt", "",
                         "IntRangeExt", "5", "-1,INT_MAX,min");
        s.end_group();
    }
    s.begin_group("g3");
    {
        s.begin_group("g4");
        {
            s.add_setting("IntListExt", "IntListExt", "",
                             "IntListExt", "1,2,3", "all");
            s.end_group();
        }
        s.add_setting("Double1", "Double1", "", "Double", "-10.0");
        s.add_setting("Double2", "Double2", "", "Double", "100.0e3");
        s.end_group();
    }

    bool ok = true;
    ASSERT_EQ(0, s.value("g1/b")->to_int(ok));
    ASSERT_TRUE(ok);
    ASSERT_TRUE(s.dependencies_satisfied("g1"));
    ASSERT_TRUE(s.dependencies_satisfied("g1/b"));
    ASSERT_FALSE(s.dependencies_satisfied("g1/d"));

    SettingsModel model(&s);
    SettingsView view(&widget);
    QVBoxLayout vlayout(&widget);
    view.setModel(&model);
    SettingsDelegate delegate(&view, &widget);
    view.setItemDelegate(&delegate);

    widget.setMinimumWidth(400);
    widget.setMinimumHeight(400);
    view.expandSettingsTree();

    vlayout.addWidget(&view);
    widget.show();
    app.exec();
    for (int i = 0; i < argc; ++i)
        free(argv[i]);
    free(argv);
}


TEST(test_SettingsModel_new, test2)
{
    int argc = 1;
    char** argv = (char**)malloc(argc * sizeof(char*));
    for (int i = 0; i < argc; ++i)
        argv[i] = (char*)malloc(255 * sizeof(char));
    QApplication app(argc, argv);
    QWidget widget;

    // Create the settings tree.
    SettingsTree s;
    settings_declare_xml(&s, oskar_XML_STR);

    // Create a file handler.
    SettingsFileHandlerQSettings file_handler;
    file_handler.set_write_defaults(false);
    s.set_file_handler(&file_handler, "test2.ini");

    s.print();

    SettingsModel model(&s);
    SettingsView view(&widget);
    QVBoxLayout vlayout(&widget);
    view.setModel(&model);
    SettingsDelegate delegate(&view, &widget);
    view.setItemDelegate(&delegate);

    widget.setMinimumWidth(800);
    widget.setMinimumHeight(800);
    view.expandSettingsTree();

    vlayout.addWidget(&view);
    widget.show();
    app.exec();
    for (int i = 0; i < argc; ++i)
        free(argv[i]);
    free(argv);
}


