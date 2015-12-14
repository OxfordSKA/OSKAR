/*
 * Copyright (c) 2013-2014, The University of Oxford
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
#include <DateTime.hpp>
#include <iostream>

TEST(DateTime, test1)
{
    using oskar::DateTime;
    DateTime t(2000, 9, 21, 5, 5, 15.1234);
    ASSERT_STREQ("2000-09-21 05:05:15.123", t.toString().c_str());
    t = DateTime(1950, 0, 0, 0, 0, 0);
    ASSERT_STREQ("1950-00-00 00:00:00.000", t.toString().c_str());
    t = DateTime(1950, 0, 0, 0, 0, 1.234567);
    ASSERT_STREQ("1950-00-00 00:00:01.235", t.toString().c_str());

    bool ok = false;
    t.set("2000-01-02 06:12:04.567", &ok);
    ASSERT_EQ(2000, t.year());
    ASSERT_EQ(1, t.month());
    ASSERT_EQ(2, t.day());
    ASSERT_EQ(6, t.hours());
    ASSERT_EQ(12, t.minutes());
    ASSERT_DOUBLE_EQ(4.567, t.seconds());

    t.clear();
    ASSERT_STREQ("0000-00-00 00:00:00.000", t.toString().c_str());

    t.set(1981,10,15,01,02,03.456);
    ASSERT_STREQ("1981-10-15 01:02:03.456", t.toString().c_str());
}
