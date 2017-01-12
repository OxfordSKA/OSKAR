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
#include "settings/oskar_settings_types.h"

using namespace oskar;

TEST(settings_types, Time)
{
    Time t;
    ASSERT_TRUE(t.init(""));
    ASSERT_STREQ("", t.get_value().c_str());

    ASSERT_TRUE(t.set_value("0.1"));
    ASSERT_STREQ("0.1", t.get_value().c_str());
    ASSERT_DOUBLE_EQ(0.1, t.to_seconds());

    ASSERT_TRUE(t.set_default("1:2:3.45678"));
    ASSERT_EQ(1, t.value().hours);
    ASSERT_EQ(2, t.value().minutes);
    ASSERT_DOUBLE_EQ(3.45678, t.value().seconds);
    ASSERT_STREQ("01:02:03.45678", t.get_default().c_str());
    ASSERT_TRUE(t.is_default());

    ASSERT_TRUE(t.set_value("01:02:23.45678"));
    ASSERT_FALSE(t.is_default());
    ASSERT_EQ(1, t.value().hours);
    ASSERT_EQ(2, t.value().minutes);
    ASSERT_DOUBLE_EQ(23.45678, t.value().seconds);
    ASSERT_STREQ("01:02:23.45678", t.get_value().c_str());

    ASSERT_TRUE(t.set_value("123.4567891011"));
    ASSERT_FALSE(t.is_default());
    ASSERT_EQ(0, t.value().hours);
    EXPECT_EQ(2, t.value().minutes);
    EXPECT_NEAR(03.4567891011, t.value().seconds, 1e-8);
    EXPECT_DOUBLE_EQ(123.4567891011, t.to_seconds());
    EXPECT_STREQ("123.4567891011", t.get_value().c_str());

    // EXPECT_STREQ("00:02:03.4567891011", t.get_value().c_str());

    ASSERT_TRUE(t.set_value("1.234567891011121e+04"));
    ASSERT_EQ(3, t.value().hours);
    ASSERT_EQ(25, t.value().minutes);
    ASSERT_NEAR(45.67891011121, t.value().seconds, 1e-8);
    ASSERT_DOUBLE_EQ(1.234567891011121e+04, t.to_seconds());
    ASSERT_STREQ("12345.67891011121", t.get_value().c_str());
    //ASSERT_STREQ("03:25:45.67891011121", t.get_value().c_str());

    ASSERT_TRUE(t.set_value("23:02:23.45678"));

}

