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
#include <iostream>
#include <climits>
#include <oskar_settings_types.hpp>

using namespace oskar;

TEST(settings_types, IntRangeExt)
{
    bool ok = false;
    IntRangeExt i;
    {
        ASSERT_STREQ("0", i.toString().c_str());
    }
    {
        i.init("0,MAX,smin", &ok);
        ASSERT_TRUE(ok);
    }
    {
        i.fromString("10", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("10", i.toString().c_str());
    }
    {
        i.fromString("-1", &ok);
        ASSERT_FALSE(ok); // fails because -1 is outside the range.
        // Despite the fail still sets to value as expected
        ASSERT_STREQ("smin", i.toString().c_str());
    }
    {
        i.fromString("smin", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("smin", i.toString().c_str());
    }
    {
        // Not allowed as 'smax' would be unable to map
        // to a unique integer value.
        i.init("0,MAX,smin,smax", &ok);
        ASSERT_FALSE(ok);
    }
    {
        i.init("0,100,smin,smax", &ok);
        ASSERT_TRUE(ok);
    }
    {
        i.fromString("10", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("10", i.toString().c_str());
    }
    {
        i.fromString("101", &ok);
        ASSERT_FALSE(ok); // fails because -1 is outside the range.
        ASSERT_STREQ("smax", i.toString().c_str());
    }
    {
        i.fromString("-1", &ok);
        ASSERT_FALSE(ok); // fails because -1 is outside the range.
        ASSERT_STREQ("smin", i.toString().c_str());
    }
    {
        i.fromString("smin", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("smin", i.toString().c_str());
    }
    {
        i.fromString("smax", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("smax", i.toString().c_str());
    }
    {
        i.init("0,10,smin", &ok);
        ASSERT_TRUE(ok);
        i.fromString("11", &ok);
        ASSERT_FALSE(ok);
        ASSERT_STREQ("0", i.toString().c_str());
    }
}
