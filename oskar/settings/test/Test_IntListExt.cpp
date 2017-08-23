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

TEST(settings_types, IntListExt)
{
    IntListExt l;
    ASSERT_TRUE(l.init("all"));
    ASSERT_STREQ("all", l.special_string());
    ASSERT_TRUE(l.set_default("all"));
    ASSERT_TRUE(l.is_extended());
    ASSERT_TRUE(l.is_default());
    ASSERT_EQ(1, l.size());
    ASSERT_EQ(0, l.values());
    ASSERT_STREQ("all", l.get_default());
    ASSERT_STREQ("all", l.get_value());
    ASSERT_TRUE(l.set_value("1,2,  3,  4"));
    ASSERT_STREQ("1,2,3,4", l.get_value());
    ASSERT_FALSE(l.is_default());
    ASSERT_FALSE(l.is_extended());
    ASSERT_EQ(4, l.size());
    ASSERT_EQ(1, l.values()[0]);
    ASSERT_EQ(2, l.values()[1]);
    ASSERT_EQ(3, l.values()[2]);
    ASSERT_EQ(4, l.values()[3]);
    ASSERT_FALSE(l.set_default("foo"));
    ASSERT_TRUE(l.set_value("2"));
    ASSERT_STREQ("2", l.get_value());
    ASSERT_EQ(1, l.size());
    ASSERT_EQ(2, l.values()[0]);
}
