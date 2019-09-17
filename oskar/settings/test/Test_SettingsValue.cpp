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
#include "settings/oskar_SettingsValue.h"
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(SettingsValue, test1)
{
    SettingsValue v;
    ASSERT_EQ(SettingsValue::UNDEF, v.type());
    ASSERT_STREQ("Undef", v.type_name());

    Bool b;
    ASSERT_TRUE(b.set_default("false"));

    ASSERT_TRUE(v.init("Bool", ""));
    ASSERT_EQ(SettingsValue::BOOL, v.type());
    ASSERT_TRUE(v.is_default());
    ASSERT_TRUE(v.set_default("false"));
    ASSERT_TRUE(v.set_value("true"));
    ASSERT_FALSE(v.is_default());
    ASSERT_STREQ("true", v.get_value());
    ASSERT_TRUE(v.set_value("false"));
    ASSERT_TRUE(v.is_default());
    ASSERT_STREQ("false", v.get_value());

    ASSERT_TRUE(v.init("DateTime", ""));
    ASSERT_EQ(SettingsValue::DATE_TIME, v.type());
    ASSERT_TRUE(v.set_value("1985-5-23T5:6:12.12345"));
    ASSERT_EQ(1985, v.get<DateTime>().value().year);
    ASSERT_EQ(DateTime::ISO, v.get<DateTime>().value().style);
    ASSERT_STREQ("", v.get_default());
    ASSERT_STREQ("1985-05-23T05:06:12.12345", v.get_value());

    ASSERT_TRUE(v.init("Double", ""));
    ASSERT_TRUE(v.set_default("2.0"));
    ASSERT_EQ(SettingsValue::DOUBLE, v.type());
    ASSERT_STREQ("Double", v.type_name());

    bool ok = false;
    ASSERT_TRUE(v.init("DoubleRangeExt", "-MAX,MAX,min,max"));
    ASSERT_TRUE(v.set_default("min"));
    ASSERT_TRUE(v.set_value("10.0"));
    ASSERT_DOUBLE_EQ(10.0, v.to_double(ok));
    ASSERT_TRUE(ok);
    ASSERT_TRUE(v.set_value("min"));
    ASSERT_DOUBLE_EQ(-DBL_MAX, v.to_double(ok));
    ASSERT_STREQ("min", v.to_string());
    ASSERT_TRUE(ok);
    ASSERT_TRUE(v.set_value("max"));
    ASSERT_DOUBLE_EQ(DBL_MAX, v.to_double(ok));
    ASSERT_STREQ("max", v.to_string());
    ASSERT_TRUE(ok);
}

