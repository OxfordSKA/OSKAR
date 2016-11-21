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

#include <OptionList.hpp>
#include <iostream>
#include <climits>
#include <vector>
#include <string>

using namespace oskar;

TEST(OptionList, test1)
{
    // Create an option list with no allowed options.
    OptionList o;
    ASSERT_FALSE(o.isSet());
    ASSERT_TRUE(o.toString().empty());
    ASSERT_TRUE(o.value().empty());
    ASSERT_EQ((size_t)0, o.options().size());
    bool ok = true;
    ASSERT_EQ(-1, o.valueIndex(&ok));
    ASSERT_FALSE(ok);

    // Setting the value should not be possible as no allowed options are
    // registered.
    ok = true;
    const char* value = "foo";
    o.fromString(value, &ok);
    ASSERT_FALSE(ok);
    std::cout << o.value() << std::endl;
    ASSERT_FALSE(o.isSet());

    ASSERT_TRUE(o.value().empty());
    ASSERT_TRUE(o.toString().empty());

    o.fromString(value, &ok);
    ASSERT_FALSE(ok);
    ASSERT_FALSE(o.isSet());
    ASSERT_TRUE(o.value().empty());
    ASSERT_TRUE(o.toString().empty());
}

TEST(OptionList, test2)
{
    std::vector<std::string> allowed;
    allowed.push_back("one");
    allowed.push_back("two");
    allowed.push_back("cat");
    bool ok = false;
    {
        const char* value = "one";
        const char* bad_value = "foo";
        OptionList o(allowed);
        ASSERT_FALSE(o.isSet());
        ASSERT_EQ((int)allowed.size(), o.num_options());
        ASSERT_EQ(allowed.size(), o.options().size());
        o.fromString(bad_value, &ok);
        ASSERT_FALSE(ok);
        ASSERT_FALSE(o.isSet());
        o.fromString(value, &ok);
        ASSERT_TRUE(ok);
        ASSERT_TRUE(o.isSet());
        ASSERT_EQ(0, o.valueIndex(&ok));
        ASSERT_TRUE(ok);
        ASSERT_STREQ(value, o.option(0).c_str());
        ASSERT_STREQ(value, o.options()[0].c_str());
    }
    {
        const char* value = "cat";
        OptionList o(allowed, value);
        ASSERT_EQ(2, o.valueIndex(&ok));
        ASSERT_TRUE(ok);
        ASSERT_TRUE(o.isSet());
        ASSERT_STREQ(value, o.value().c_str());
        ASSERT_STREQ(value, o.toString().c_str());
        o.fromString("t", &ok);
        ASSERT_TRUE(ok);
        ASSERT_STREQ("two", o.value().c_str());
    }

}

TEST(OptionList, test3)
{
    // Note the CSV list is trimmed of white-space.
    const char* allowed = "one, two,  three,four";
    const char* value = "two";
    bool ok = false;
    OptionList o(allowed, value);
    ASSERT_TRUE(o.isSet());
    ASSERT_EQ(1, o.valueIndex(&ok));
    ASSERT_TRUE(ok);
    ASSERT_STREQ(value, o.value().c_str());
    ASSERT_EQ(4, o.num_options());
    ASSERT_STREQ("one", o.option(0).c_str());
    ASSERT_STREQ("two", o.option(1).c_str());
    ASSERT_STREQ("three", o.option(2).c_str());
    ASSERT_STREQ("four", o.option(3).c_str());
    o.fromString("four", &ok);
    ASSERT_TRUE(ok);
    ASSERT_STREQ("four", o.toString().c_str());
}
