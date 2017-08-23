/*
 * Copyright (c) 2015-2017, The University of Oxford
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
#include <cstdio>

using namespace std;
using namespace oskar;

TEST(settings_types, DateTime)
{
    /*
     *  d-M-yyyy h:m:s[.z] - British style
     *  yyyy/M/d/h:m:s[.z] - CASA style
     *  yyyy-M-d h:m:s[.z] - International style
     *  yyyy-M-dTh:m:s[.z] - ISO date style
     *  MJD
     */
    DateTime t;
    // British style
    {
        EXPECT_TRUE(t.set_default("1-2-2015 10:05:23.2"));
        EXPECT_TRUE(t.is_default());
        EXPECT_STREQ("01-02-2015 10:05:23.2", t.get_value());
    }
    // CASA style
    {
        EXPECT_TRUE(t.set_value("2015/1/2/03:04:05.6"));
        EXPECT_FALSE(t.is_default());
        EXPECT_STREQ("2015/01/02/03:04:05.6", t.get_value());
    }
    // International style
    {
        EXPECT_TRUE(t.set_value("2015-2-3 04:05:06.7"));
        EXPECT_FALSE(t.is_default());
        EXPECT_DOUBLE_EQ(6.7, t.value().seconds);
        EXPECT_DOUBLE_EQ(23.2, t.default_value().seconds);
        EXPECT_STREQ("2015-02-03 04:05:06.7", t.get_value());
    }
    // International style
    {
        EXPECT_TRUE(t.set_value("2015-12-31 23:59:59.0"));
        EXPECT_FALSE(t.is_default());
        EXPECT_DOUBLE_EQ(59.0, t.value().seconds);
        EXPECT_EQ(59, t.value().minutes);
        EXPECT_EQ(23, t.value().hours);
        EXPECT_EQ(31, t.value().day);
        EXPECT_EQ(12, t.value().month);
        EXPECT_EQ(2015, t.value().year);
        EXPECT_DOUBLE_EQ(23.2, t.default_value().seconds);
        EXPECT_STREQ("2015-12-31 23:59:59.0", t.get_value());
    }
    // ISO style
    {
        EXPECT_TRUE(t.set_value("2015-3-4T05:06:07.8910111213"));
        EXPECT_FALSE(t.is_default());
    }
    // MJD1
    {
        double mjd = t.to_mjd();
        double mjd2 = t.to_mjd_2();
        EXPECT_DOUBLE_EQ(mjd, mjd2);
        t.from_mjd(mjd);
        EXPECT_DOUBLE_EQ(mjd, t.to_mjd());

    }
    // MJD2
    {
        double mjd = 46113.7511111;
        t.from_mjd(mjd);
        EXPECT_DOUBLE_EQ(mjd, t.to_mjd());
        EXPECT_EQ(DateTime::MJD, t.format());
    }
    {
        DateTime t1;
        EXPECT_TRUE(t1.set_value("46113.7654321"));
        EXPECT_EQ(DateTime::MJD, t1.format());
        EXPECT_STREQ("46113.7654321", t1.get_value());
        EXPECT_DOUBLE_EQ(46113.7654321, t1.to_mjd());
    }
    {
        DateTime t1;
        t1.set_value("45464844.54646541");
        EXPECT_STREQ("45464844.546465", t1.get_value());
    }

}
