/*
 * Copyright (c) 2014, The University of Oxford
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
#include <climits>

using namespace oskar;

TEST(settings_types, IntRangeExt)
{
    IntRangeExt i;
    ASSERT_STREQ("0", i.get_default());
    ASSERT_TRUE(i.init("0,MAX,smin"));
    ASSERT_EQ(0, i.min());
    ASSERT_EQ(INT_MAX, i.max());
    ASSERT_STREQ("smin", i.ext_min());
    ASSERT_TRUE(i.set_default("12"));
    ASSERT_EQ(12, i.value());
    ASSERT_STREQ("12", i.get_default());
    ASSERT_STREQ("12", i.get_value());
    ASSERT_TRUE(i.is_default());
    ASSERT_FALSE(i.set_value("-1"));
    ASSERT_STREQ("smin", i.get_value());
    ASSERT_FALSE(i.is_default());
    ASSERT_TRUE(i.set_value("10"));
    ASSERT_STREQ("10", i.get_value());
    // Not allowed as 'smax' would be unable to map
    // to a unique integer value.
    ASSERT_FALSE(i.init("0,MAX,smin,smax"));
    ASSERT_TRUE(i.init("0,100,smin,smax"));
    ASSERT_TRUE(i.set_default("10"));
    ASSERT_STREQ("10", i.get_default());
    ASSERT_FALSE(i.set_value("101")); // fails because -1 is outside the range.
    ASSERT_STREQ("smax", i.get_value());
    ASSERT_FALSE(i.set_value("-1")); // fails because -1 is outside the range.
    ASSERT_STREQ("smin", i.get_value());
    ASSERT_TRUE(i.set_value("smin"));
    ASSERT_STREQ("smin", i.get_value());
    ASSERT_TRUE(i.set_value("smax"));
    ASSERT_STREQ("smax", i.get_value());
    ASSERT_TRUE(i.init("0,10,smin"));
    ASSERT_FALSE(i.set_value("11"));
    ASSERT_STREQ("0", i.get_value());

    ASSERT_TRUE(i.init("-1,INT_MAX,smin"));
    ASSERT_TRUE(i.set_default("5"));
    ASSERT_EQ(5, i.value());
    ASSERT_TRUE(i.set_value("smin"));
    ASSERT_STREQ("smin", i.get_value());
}
