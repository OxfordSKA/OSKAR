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
using namespace std;

TEST(settings_types, DoubleRangeExt)
{
    DoubleRangeExt r;
    {
        // This should fail as there is no extended string
        ASSERT_FALSE(r.init("2.0,5.0"));
        ASSERT_STREQ("0.0", r.get_default());
    }
    {
        ASSERT_TRUE(r.init("2.0,5.0,min"));
        ASSERT_STREQ("0.0", r.get_default());
        ASSERT_TRUE(r.set_default("3.2"));
        ASSERT_TRUE(r.is_default());
        // FIXME(BM) double to string printing...
        EXPECT_STREQ("3.2", r.get_default());
        EXPECT_STREQ("3.2", r.get_value());
        ASSERT_DOUBLE_EQ(3.2, r.value());
        ASSERT_TRUE(r.set_value("5.1"));
        ASSERT_STREQ("5.0", r.get_value());
        ASSERT_TRUE(r.set_value("1.1"));
        ASSERT_STREQ("min", r.get_value());
        ASSERT_TRUE(r.set_value("2.1234567891"));
        ASSERT_STREQ("2.1234567891", r.get_value());
    }
    {
        ASSERT_TRUE(r.init("2.0,5.0,min, max"));
        ASSERT_STREQ("0.0", r.get_value());
        ASSERT_STREQ("0.0", r.get_default());
        ASSERT_TRUE(r.set_value("3.0"));
        ASSERT_STREQ("3.0", r.get_value());
        ASSERT_TRUE(r.set_value("3.53"));
        EXPECT_STREQ("3.53", r.get_value());
        ASSERT_DOUBLE_EQ(3.53, r.value());
        ASSERT_TRUE(r.set_value("5.1"));
        ASSERT_STREQ("max", r.get_value());
        ASSERT_TRUE(r.set_value("1.1"));
        ASSERT_STREQ("min", r.get_value());
        ASSERT_TRUE(r.set_value("2.1234567891"));
        ASSERT_STREQ("2.1234567891", r.get_value());
    }
    {
        ASSERT_TRUE(r.init("MIN,MAX,MAX"));
    }
    {
        ASSERT_TRUE(r.init("-MAX,MAX,min,max"));
        ASSERT_TRUE(r.set_default("min"));
        ASSERT_STREQ("min", r.get_value());
    }
    {
        ASSERT_TRUE(r.init("-MAX,MAX,min,max"));
        ASSERT_TRUE(r.set_default("max"));
        ASSERT_STREQ("max", r.get_value());
        r.set_value("10.0");
        ASSERT_STREQ("10.0", r.get_value());
    }
}

